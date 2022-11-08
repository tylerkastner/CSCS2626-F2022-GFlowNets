import torch
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader

# example inputs and outputs
# inputs = [[1,0,0,0,1,0,0,0,2]]
# targets = [2.6]

df = pd.read_csv('old/samples_trajs.csv', sep=',', header=None)

horizon = 8
ndim = 2

inputs = df.to_numpy()[:,:horizon*ndim+1]
targets = df.to_numpy()[:,horizon*ndim+2]

inps = torch.tensor(inputs, dtype=torch.int8)
tgts = torch.tensor(targets, dtype=torch.float32)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)