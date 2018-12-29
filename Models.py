import torch
from torch import nn

from torch.optim import RMSprop as RMS

# global declarations

hm_channels = 3
stride      = 1

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


# models


class Generator(nn.Module):
    def __init__(self, noise_size, layers, filters, width, height):
        super(Generator, self).__init__()
        self.noise_size = noise_size

        self.model = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels  = 1,
                out_channels = filters[0],
                kernel_size  = (required_kernel_size(layers[0], noise_size),
                                required_kernel_size(layers[0], noise_size)),
                stride       = stride,
                bias         = False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels  = filters[0],
                out_channels = filters[1],
                kernel_size  = (required_kernel_size(layers[1], layers[0]),
                                required_kernel_size(layers[1], layers[0])),
                stride       = stride,
                bias         = False),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=filters[1],
                out_channels=filters[2],
                kernel_size=(required_kernel_size(layers[2], layers[1]),
                             required_kernel_size(layers[2], layers[1])),
                stride=stride,
                bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(),

            # nn.ConvTranspose2d(
            #     in_channels=filters[2],
            #     out_channels=filters[3],
            #     kernel_size=(required_kernel_size(layers[3], layers[2]),
            #                  required_kernel_size(layers[3], layers[2])),
            #     stride=stride,
            #     bias=False),
            # nn.BatchNorm2d(filters[3]),
            # nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels  = filters[-1],
                out_channels = hm_channels,
                kernel_size  = (required_kernel_size(width, layers[-1]),
                                required_kernel_size(height, layers[-1])),
                stride       = stride,
                bias         = False),
            nn.BatchNorm2d(hm_channels),
            nn.Tanh(),
        )

    def forward(self, batchsize=1): return self.model(torch.randn(batchsize, 1, self.noise_size, self.noise_size).to('cuda'))


class Discriminator(nn.Module):
    def __init__(self, width, height, layers, filters):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(
                in_channels  = hm_channels,
                out_channels = filters[0],
                kernel_size  = (required_kernel_size(width, layers[0]),
                                required_kernel_size(height, layers[0])),
                stride       = stride,
                bias         = False),
            nn.BatchNorm2d(filters[0]),
            nn.LeakyReLU(),

            # nn.Conv2d(
            #     in_channels=filters[-5],
            #     out_channels=filters[-4],
            #     kernel_size=(required_kernel_size(layers[-5], layers[-4]),
            #                  required_kernel_size(layers[-5], layers[-4])),
            #     stride=stride,
            #     bias=False),
            # nn.BatchNorm2d(filters[-4]),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(
            #     in_channels=filters[-4],
            #     out_channels=filters[-3],
            #     kernel_size=(required_kernel_size(layers[-4], layers[-3]),
            #                  required_kernel_size(layers[-4], layers[-3])),
            #     stride=stride,
            #     bias=False),
            # nn.BatchNorm2d(filters[-3]),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(
            #     in_channels=filters[-3],
            #     out_channels=filters[-2],
            #     kernel_size=(required_kernel_size(layers[-3], layers[-2]),
            #                  required_kernel_size(layers[-3], layers[-2])),
            #     stride=stride,
            #     bias=False),
            # nn.BatchNorm2d(filters[-2]),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(
            #     in_channels  = filters[-2],
            #     out_channels = filters[-1],
            #     kernel_size  = (required_kernel_size(layers[-2], layers[-1]),
            #                     required_kernel_size(layers[-2], layers[-1])),
            #     stride       = stride,
            #     bias         = False),
            # nn.BatchNorm2d(filters[-1]),
            # nn.LeakyReLU(),


            nn.Conv2d(
                in_channels=filters[-1],
                out_channels=1,
                kernel_size=(required_kernel_size(layers[-1], 1),
                             required_kernel_size(layers[-1], 1)),
                stride=stride,
                bias=False),
            nn.Sigmoid(),
        )

        # flat_size = layers[-1] * layers[-1] * filters[-1]

        # self.model2 = nn.Sequential(
        #
        #     nn.Linear(flat_size, 1, bias=False),
        #     nn.Sigmoid()
        # )

    def forward(self, inp):
        result_pre = self.model(inp)
        # result = self.model2(result_pre.view(result_pre.size(0), -1))
        return result_pre.squeeze(-1).squeeze(-1).sum(-1)  # .view(result_pre.size(0))



def loss_discriminator(discriminator_results, labels):
    return - (labels * torch.log(discriminator_results) + (1-labels) * torch.log(1-discriminator_results)).sum()

def loss_discriminator_w(real_results, fake_results):
    return - (real_results - fake_results).sum()

def loss_generator(discriminator_results, type='minimize'):
    if type == 'minimize':
        return - (torch.log(discriminator_results)).sum()
    else: return - (torch.log(1-discriminator_results)).sum()

optimizers = []

def update(loss, discriminator, generator, update_for, maximize_loss=False, lr=0.001, batch_size=1):
    global optimizers
    if not optimizers: optimizers = (RMS(discriminator.parameters(), lr), RMS(generator.parameters(), lr))

    loss /= batch_size
    loss.backward()

    if maximize_loss:
        for param in generator.parameters():
            if param.grad is not None:
                param.grad = -param.grad
            else:
                print(f'none grad on {update_for} : generator')

    with torch.no_grad():

        if update_for == 'discriminator':

            optimizers[0].step()
            optimizers[0].zero_grad()

            for param in generator.parameters():
                if param.grad is not None:
                    param.grad = None
                else: print(f'none grad on {update_for} : generator')

        elif update_for == 'generator':

            for param in discriminator.parameters():
                if param.grad is not None:
                    param.grad = None
                else: print(f'none grad on {update_for} : discriminator')

            optimizers[1].step()
            optimizers[1].zero_grad()


def interact(generator): return generator.forward()


# helpers


def required_kernel_size(this_size, target_size): return this_size - (target_size-1) * stride
