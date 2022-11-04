import torch as torch
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt
import itertools
from scipy.stats import norm

device = torch.device('cpu')

horizon = 8
ndim = 2

n_hid = 256
n_layers = 2

bs = 16

detailed_balance = False  # else, traj balance
uniform_pb = False

print('loss is', 'DB' if detailed_balance else 'TB')



def currin(x):
    x_0 = x[..., 0] / 2 + 0.5
    x_1 = x[..., 1] / 2 + 0.5
    factor1 = 1 - np.exp(- 1 / (2 * x_1 + 1e-10))
    numer = 2300 * x_0 ** 3 + 1900 * x_0 ** 2 + 2092 * x_0 + 60
    denom = 100 * x_0 ** 3 + 500 * x_0 ** 2 + 4 * x_0 + 20
    return factor1 * numer / denom / 13.77 # Dividing by the max to help normalize

def branin(x):
    x_0 = 15 * (x[..., 0] / 2 + 0.5) - 5
    x_1 = 15 * (x[..., 1] / 2 + 0.5)
    t1 = (x_1 - 5.1 / (4 * np.pi ** 2) * x_0 ** 2
          + 5 / np.pi * x_0 - 6)
    t2 = 10 * (1 - 1 / (8 * np.pi)) * np.cos(x_0)
    return 1 - (t1 ** 2 + t2 + 10) / 308.13 # Dividing by the max to help normalize

class GridEnv:

    def __init__(self, horizon, ndim=2, xrange=[-1, 1], funcs=None,
                 obs_type='one-hot'):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.funcs = ([lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01] if funcs is None else funcs)
        self.num_cond_dim = len(self.funcs) + 1
        self.xspace = np.linspace(*xrange, horizon)
        self._true_density = None
        self.obs_type = obs_type
        if obs_type == 'one-hot':
            self.num_obs_dim = self.horizon * self.ndim
        elif obs_type == 'scalar':
            self.num_obs_dim = self.ndim
        elif obs_type == 'tab':
            self.num_obs_dim = self.horizon ** self.ndim

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros(self.num_obs_dim + self.num_cond_dim)
        if self.obs_type == 'one-hot':
            z = np.zeros((self.horizon * self.ndim + self.num_cond_dim), dtype=np.float32)
            z[np.arange(len(s)) * self.horizon + s] = 1
        elif self.obs_type == 'scalar':
            z[:self.ndim] = self.s2x(s)
        elif self.obs_type == 'tab':
            idx = (s * (self.horizon ** np.arange(self.ndim))).sum()
            z[idx] = 1
        z[-self.num_cond_dim:] = self.cond_obs
        return z

    def s2x(self, s):
        return s / (self.horizon - 1) * self.width + self.start

    def s2r(self, s):
        x = self.s2x(s)
        return (self.coefficients * np.array([i(x) for i in self.funcs])).sum() ** self.temperature

    def reset(self, coefs=None, temp=None):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        self.coefficients = np.random.dirichlet([1.5] * len(self.funcs)) if coefs is None else coefs
        self.temperature = np.random.gamma(2, 1) if temp is None else temp
        self.cond_obs = np.concatenate([self.coefficients, [self.temperature]])
        return self.obs(), self.s2r(self._state), self._state

    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.s2r(s), done, s

    def state_info(self):
        all_int_states = np.float32(list(itertools.product(*[list(range(self.horizon))] * self.ndim)))
        state_mask = (all_int_states == self.horizon - 1).sum(1) <= 1
        pos = all_int_states[state_mask].astype('float')
        s = pos / (self.horizon - 1) * (self.xspace[-1] - self.xspace[0]) + self.xspace[0]
        r = self.funcs(s)
        return s, r, pos

    def render(self):

        plt.matshow()









def make_mlp(l, act=torch.nn.LeakyReLU(), tail=[]):
    return torch.nn.Sequential(*(sum(
        [[torch.nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))


def log_reward(x):
    ax = abs(x / (horizon - 1) * 2 - 1)
    return ((ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-3).log()


j = torch.zeros((horizon,) * ndim + (ndim,))
for i in range(ndim):
    jj = torch.linspace(0, horizon - 1, horizon)
    for _ in range(i): jj = jj.unsqueeze(1)
    j[..., i] = jj

truelr = log_reward(j)
print('total reward', truelr.view(-1).logsumexp(0))
true_dist = truelr.flatten().softmax(0).cpu().numpy()
plt.matshow(truelr)
plt.show()

def toin(z):
    return torch.nn.functional.one_hot(z, horizon).view(z.shape[0], -1).float()


Z = torch.zeros((1,)).to(device)
if detailed_balance:
    model = make_mlp([ndim * horizon] + [n_hid] * n_layers + [2 * ndim + 2]).to(device)
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
else:
    model = make_mlp([ndim * horizon] + [n_hid] * n_layers + [2 * ndim + 1]).to(device)
    opt = torch.optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': [Z], 'lr': 0.1}])
    Z.requires_grad_()

losses = []
zs = []
all_visited = []
first_visit = -1 * np.ones_like(true_dist)
l1log = []

for it in tqdm.trange(62501):
    opt.zero_grad()

    z = torch.zeros((bs, ndim), dtype=torch.long).to(device)
    done = torch.full((bs,), False, dtype=torch.bool).to(device)

    action = None

    if detailed_balance:
        ll_diff = torch.zeros((ndim * horizon, bs)).to(device)
    else:
        ll_diff = torch.zeros((bs,)).to(device)
        ll_diff += Z

    i = 0
    while torch.any(~done):

        pred = model(toin(z[~done]))

        edge_mask = torch.cat([(z[~done] == horizon - 1).float(), torch.zeros(((~done).sum(), 1), device=device)], 1)
        logits = (pred[..., :ndim + 1] - 1000000000 * edge_mask).log_softmax(1)

        init_edge_mask = (z[~done] == 0).float()
        back_logits = ((0 if uniform_pb else 1) * pred[..., ndim + 1:2 * ndim + 1] - 1000000000 * init_edge_mask).log_softmax(1)

        if detailed_balance:
            log_flow = pred[..., 2 * ndim + 1]
            ll_diff[i, ~done] += log_flow
            if i > 0:
                ll_diff[i - 1, ~done] -= log_flow
            else:
                Z[:] = log_flow[0].item()

        if action is not None:
            if detailed_balance:
                ll_diff[i - 1, ~done] -= back_logits.gather(1, action[action != ndim].unsqueeze(1)).squeeze(1)
            else:
                ll_diff[~done] -= back_logits.gather(1, action[action != ndim].unsqueeze(1)).squeeze(1)

        exp_weight = 0.
        temp = 1
        sample_ins_probs = (1 - exp_weight) * (logits / temp).softmax(1) + exp_weight * (1 - edge_mask) / (
                    1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)

        action = sample_ins_probs.multinomial(1)
        if detailed_balance:
            ll_diff[i, ~done] += logits.gather(1, action).squeeze(1)
        else:
            ll_diff[~done] += logits.gather(1, action).squeeze(1)

        terminate = (action == ndim).squeeze(1)
        for x in z[~done][terminate]:
            state = (x.cpu() * (horizon ** torch.arange(ndim))).sum().item()
            if first_visit[state] < 0: first_visit[state] = it
            all_visited.append(state)

        if detailed_balance:
            termination_mask = ~done
            termination_mask[~done] &= terminate
            ll_diff[i, termination_mask] -= log_reward(z[~done][terminate].float())
        done[~done] |= terminate

        with torch.no_grad():
            z[~done] = z[~done].scatter_add(1, action[~terminate], torch.ones(action[~terminate].shape, dtype=torch.long, device=device))

        i += 1

    lens = z.sum(1) + 1
    if not detailed_balance:
        lr = log_reward(z.float())
        ll_diff -= lr

    loss = (ll_diff ** 2).sum() / (lens.sum() if detailed_balance else bs)

    loss.backward()

    opt.step()

    losses.append(loss.item())

    zs.append(Z.item())

    if it % 1000 == 0:
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-200000:], minlength=len(true_dist)).astype(float)
        emp_dist /= emp_dist.sum()
        l1 = np.abs(true_dist - emp_dist).mean()
        print('L1 =', l1)
        l1log.append((len(all_visited), l1))
        emp_dist =emp_dist.reshape(horizon, horizon)
        plt.matshow(emp_dist)
        plt.show()

pickle.dump([losses, zs, all_visited, first_visit, l1log], open(f'out.pkl', 'wb'))