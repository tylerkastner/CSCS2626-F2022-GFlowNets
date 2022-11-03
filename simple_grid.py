import torch as T
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt

device = T.device('cpu')

horizon = 8
ndim = 2

n_hid = 256
n_layers = 2

bs = 16

detailed_balance = False  # else, traj balance
uniform_pb = False

print('loss is', 'DB' if detailed_balance else 'TB')


def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))


def log_reward(x):
    ax = abs(x / (horizon - 1) * 2 - 1)
    return ((ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-3).log()


j = T.zeros((horizon,) * ndim + (ndim,))
for i in range(ndim):
    jj = T.linspace(0, horizon - 1, horizon)
    for _ in range(i): jj = jj.unsqueeze(1)
    j[..., i] = jj

truelr = log_reward(j)
print('total reward', truelr.view(-1).logsumexp(0))
true_dist = truelr.flatten().softmax(0).cpu().numpy()
plt.imshow(truelr)
plt.show()

def toin(z):
    return T.nn.functional.one_hot(z, horizon).view(z.shape[0], -1).float()


Z = T.zeros((1,)).to(device)
if detailed_balance:
    model = make_mlp([ndim * horizon] + [n_hid] * n_layers + [2 * ndim + 2]).to(device)
    opt = T.optim.Adam([{'params': model.parameters(), 'lr': 0.001}])
else:
    model = make_mlp([ndim * horizon] + [n_hid] * n_layers + [2 * ndim + 1]).to(device)
    opt = T.optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': [Z], 'lr': 0.1}])
    Z.requires_grad_()

losses = []
zs = []
all_visited = []
first_visit = -1 * np.ones_like(true_dist)
l1log = []

for it in tqdm.trange(62501):
    opt.zero_grad()

    z = T.zeros((bs, ndim), dtype=T.long).to(device)
    done = T.full((bs,), False, dtype=T.bool).to(device)

    action = None

    if detailed_balance:
        ll_diff = T.zeros((ndim * horizon, bs)).to(device)
    else:
        ll_diff = T.zeros((bs,)).to(device)
        ll_diff += Z

    i = 0
    while T.any(~done):

        pred = model(toin(z[~done]))

        edge_mask = T.cat([(z[~done] == horizon - 1).float(), T.zeros(((~done).sum(), 1), device=device)], 1)
        logits = (pred[..., :ndim + 1] - 1000000000 * edge_mask).log_softmax(1)

        init_edge_mask = (z[~done] == 0).float()
        back_logits = ((0 if uniform_pb else 1) * pred[...,
                                                  ndim + 1:2 * ndim + 1] - 1000000000 * init_edge_mask).log_softmax(1)

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
            state = (x.cpu() * (horizon ** T.arange(ndim))).sum().item()
            if first_visit[state] < 0: first_visit[state] = it
            all_visited.append(state)

        if detailed_balance:
            termination_mask = ~done
            termination_mask[~done] &= terminate
            ll_diff[i, termination_mask] -= log_reward(z[~done][terminate].float())
        done[~done] |= terminate

        with T.no_grad():
            z[~done] = z[~done].scatter_add(1, action[~terminate],
                                            T.ones(action[~terminate].shape, dtype=T.long, device=device))

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

    if it % 100 == 0:
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-200000:], minlength=len(true_dist)).astype(float)
        emp_dist /= emp_dist.sum()
        l1 = np.abs(true_dist - emp_dist).mean()
        print('L1 =', l1)
        l1log.append((len(all_visited), l1))
        emp_dist =emp_dist.reshape(horizon, horizon)

pickle.dump([losses, zs, all_visited, first_visit, l1log], open(f'out.pkl', 'wb'))