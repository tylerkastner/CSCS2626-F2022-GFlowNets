import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import torch

def render_distribution(distribution, box_height, ndim, file_name):
    '''
        Renders either the given distribution or the env grountruth distribution
    '''

    Path('./results/').mkdir(parents=True, exist_ok=True)
    if ndim == 2:
        plt.matshow(distribution.T)
        plt.title(file_name)
        plt.show()
    elif ndim == 3:
        X, Y, Z = np.mgrid[0:box_height:1, 0:box_height:1, 0:box_height:1]
        fig = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=distribution.flatten(),
            isomin=0.01,
            isomax=0.8,
            opacity=0.1,  # needs to be small to see through all surfaces
            surface_count=20,  # needs to be a large number for good volume rendering
        ))
        fig.write_html('./results/{}.html'.format(file_name))
    else:
        print('Rendering for {} dimension not supported.'.format(ndim))



def frequency_features(t, n_features):
    features = [np.sin(t * 2 * np.pi * n) for n in range(1,n_features+1)]
    return torch.tensor(features)
