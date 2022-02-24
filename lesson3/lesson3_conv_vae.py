import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy


#hyper params
num_epoch = 20
cuda_device = -1
batch_size = 128
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

#model


class Encoder(nn.Module):
    # 28*28 -> hidden -> out
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        #conv2d -> maxpool2d -> conv2d -> maxpool2d -> conv2d
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.pooling1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pooling2 = nn.MaxPool2d((2, 2))
        #TODO поробовать линейниы слои для мю и сигмы. После 3й свертки добавить 2 линеара
        self.conv_mu = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.conv_sigma = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.pooling1(self.activation(self.conv1(x)))
        x = self.pooling2(self.activation(self.conv2(x)))
        mu = self.conv_mu(x)
        sigma = self.conv_sigma(x)
        # 7*7
        return mu, torch.exp(sigma)


def sampling(mu, sigma):
    return mu + torch.normal(torch.zeros_like(sigma), torch.ones_like(sigma)) * sigma


class Dencoder(nn.Module):
    #conv2d -> upsampling2d -> conv2d -> upsampling2d -> conv2d
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=5, stride=1, padding=2)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.upsampling1(self.activation(self.conv1(x)))
        x = self.upsampling2(self.activation(self.conv2(x)))
        x = self.activation(self.conv3(x))

        return x


class AutoEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 latent_space: int,
                 ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_space)
        self.decoder = Dencoder(latent_space, hidden_dim, input_dim)


    def forward(self, x):
        mu, sigma = self.encoder(x)
        x = sampling(mu, sigma)
        x = self.decoder(x)

        return x, mu, sigma


def collate_fn(data):
    pics = []
    target = []
    for item in data:

        pics.append(numpy.array(item[0]))
        target.append(item[1])
    return {
        'data': torch.from_numpy(numpy.array(pics)).float() / 255,
        'target': torch.from_numpy(numpy.array(target)),
    }


def kl_loss(mu, sigma):
    p = torch.distributions.Normal(mu, sigma)
    q = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    return torch.distributions.kl_divergence(p, q)


# test_tersor = torch.ones([1, 28*28])
model = AutoEncoder(input_dim=28*28, hidden_dim=256, latent_space=64)
model.train()
model.to(device)
# result = model(test_tersor)

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler

#dataset
dataset = datasets.MNIST('.', download=False)

#
# import matplotlib.pyplot as plt
# plt.imshow(dataset.data[0].detach().numpy())
# plt.show()
#loss
loss_func = nn.MSELoss()
#dataloder

for epoch in range(2):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        data = batch['data'].to(device).unsqueeze(1)
        data_noized = torch.clamp(data + torch.normal(torch.zeros_like(data), torch.ones_like(data)), 0., 1.)
        optim.zero_grad()
        predict, mu, sigma = model(data_noized)
        loss = loss_func(predict, data) + kl_loss(mu, sigma)
        loss.backward()
        optim.step()
        if (step % 50 == 0):
            print(loss)
    print(f'epoch: {epoch}')


# test = dataset.data[65].unsqueeze(0).unsqueeze(0).float() / 255
# predict = model(test)
# import matplotlib.pyplot as plt
# plt.imshow(predict[0][0].detach().numpy())
# plt.show()
#
# plt.imshow(test[0].view(28,28).detach().numpy())
# plt.show()