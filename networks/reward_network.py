import torch.nn as nn

class RewardNetwork(nn.Module):
  def __init__(
    self,
    state_dim,
    state_height,
    hidden_dim1 = 128,
    out_features = 1,
  ):
    self.state_height = state_height
    super(RewardNetwork, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(state_dim, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, hidden_dim1),
      nn.ReLU(),
      nn.Linear(hidden_dim1, out_features),
    )
    # self.net = nn.Sequential(
    #   nn.Linear(state_dim, out_features)
    # )

  def forward(self, x):
    x = x/(self.state_height-1) - 0.5
    return self.net(x)
