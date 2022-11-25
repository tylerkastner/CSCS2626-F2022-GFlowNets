import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import numpy as np
import pathlib
from utils import frequency_features

number = 0
data_path = './data/MNIST/{}/'.format(number)
generate_data = False
show_data = False

if generate_data:
    c = 0
    training_data = datasets.MNIST(root="./data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=ToTensor())

    path = pathlib.Path(data_path)
    path.mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(training_data):
        img, label = d
        if label == number:
            np.save('{}/{}'.format(data_path, c), img)
            c += 1

    for k, d in enumerate(test_data):
        img, label = d
        if label == 0:
            np.save('{}/{}'.format(data_path, c), img)
            c += 1

if show_data:
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        img = np.load('{}{}.npy'.format(data_path, i))
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()



class MNISTTrajLoader(Dataset):
    def __init__(self, img_dir='./data/MNIST/0/', n_noise_steps=50, noise_strategy='uniform', eps_noise_background=1.0, beta=0.1, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.n_noising_steps = n_noise_steps
        self.eps_background_noise = eps_noise_background / self.n_noising_steps
        self.noise_strategy = noise_strategy
        self.beta = beta
        self.dx = 1 / self.n_noising_steps
        self.time_index = np.arange(0,self.n_noising_steps, 1)

    def __len__(self):
        file_names = os.listdir(self.img_dir)
        return len(file_names)

    def __getitem__(self, idx):
        img = np.load('{}{}.npy'.format(self.img_dir, idx))

        if self.noise_strategy == 'uniform':
            img = img.squeeze()
            noise = np.random.uniform(0, 1, size=(self.n_noising_steps, img.shape[0], img.shape[1]))
            noise_length = noise.sum(0, keepdims=True)
            noise = noise / noise_length * img
            background_noise = np.random.uniform(0, self.eps_background_noise, size=(self.n_noising_steps, img.shape[0], img.shape[1]))

            image_trajectory = np.cumsum(noise, axis=0)
            background_noise_trajectory = np.cumsum(background_noise, axis=0)
            image_trajectory = image_trajectory + background_noise_trajectory
        elif self.noise_strategy == 'gaussian':
            image_trajectory = np.repeat(img, self.n_noising_steps, axis=0)
            noise = np.random.normal(0, self.beta, size=(self.n_noising_steps, img.shape[1], img.shape[2]))

            for i in range(1, self.n_noising_steps):
                image_trajectory[i] = noise[i-1] + np.sqrt(1- self.beta) * image_trajectory[i-1]
        elif self.noise_strategy == 'discrete_uniform':
            img = img.squeeze()
            n_target_steps = self.n_noising_steps * img
            n_target_steps = np.array(np.round(n_target_steps,).reshape(-1), dtype=np.int8)
            noise = np.zeros((self.n_noising_steps, img.shape[0], img.shape[1])).reshape(self.n_noising_steps, -1)
            for i in range(noise.shape[-1]):
                idx = np.random.choice(self.n_noising_steps, size=(n_target_steps[i]), replace=False)
                noise[idx, i] = 1
            noise = noise.reshape(self.n_noising_steps, img.shape[0], img.shape[1])

            image_trajectory = np.cumsum(noise, axis=0) * self.dx




        return image_trajectory, noise

if __name__ == '__main__':
    custom_mnist_dataset = MNISTTrajLoader(eps_noise_background=0.1, noise_strategy='discrete_uniform', beta=0.15)
    train_dataloader = DataLoader(custom_mnist_dataset, batch_size=64, shuffle=True)

    traj, actions = next(iter(train_dataloader))

    first_traj = traj[0]
    for i, img in enumerate(first_traj):
        if i % 1 == 0:
            plt.imshow(img, vmin=0, vmax=1)
            plt.show()
