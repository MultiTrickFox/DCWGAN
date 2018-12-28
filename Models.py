import torch
from torch import nn

from multiprocessing.pool import ThreadPool as Pool


# global declarations

hm_channels = 3
stride      = 1


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
                out_channels = filters[-1],
                kernel_size  = (required_kernel_size(layers[-1], layers[0]),
                                required_kernel_size(layers[-1], layers[0])),
                stride       = stride,
                bias         = False),
            nn.BatchNorm2d(filters[-1]),
            nn.ReLU(),

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

            nn.Conv2d(
                in_channels  = filters[0],
                out_channels = filters[-1],
                kernel_size  = (required_kernel_size(layers[0], layers[-1]),
                                required_kernel_size(layers[0], layers[-1])),
                stride       = stride,
                bias         = False),
            nn.BatchNorm2d(filters[-1]),
            nn.LeakyReLU(),
        )

        flat_size = layers[-1] * layers[-1] * filters[-1]

        self.model2 = nn.Sequential(

            nn.Linear(flat_size, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        result_pre = self.model(inp)
        result = self.model2(result_pre.view(result_pre.size(0), -1))
        return result



def loss_discriminator(discriminator_results, labels):
    with Pool(len(labels)) as p:
        results = p.map(d_loss_fn, tuple([(discriminator_result, label) for discriminator_result, label in zip(discriminator_results, labels)]))

        p.close()
        p.join()

    return sum(results)

def d_loss_fn(data):
    output, target = data
    if target == 1:
        return (- torch.log(output)).sum()
    else:
        return (- torch.log(1 - output)).sum()


def loss_generator(discriminator_results, loss_type='minimize'):
    with Pool(len(discriminator_results)) as p:
        results = p.map(d_loss_fn, tuple([(discriminator_result, loss_type) for discriminator_result in discriminator_results]))

        p.close()
        p.join()

    return sum(results)

def g_loss_fn(data):
    output, type = data
    if type == 'minimize':
        return (- torch.log(output)).sum()
    else:
        return (- torch.log(1 - output)).sum()



def update(loss, discriminator, generator, update_for, maximize_loss=False, lr=0.001, batch_size=1):
    loss.backward()
    discriminator = discriminator.parameters()
    generator = generator.parameters()
    with torch.no_grad():

        if update_for == 'discriminator':

            for param in discriminator:
                if param.grad is not None:
                    param -= lr * param.grad / batch_size
                    param.grad = None
                else: print(f'none grad on {update_for} : discriminator')

            for param in generator:
                if param.grad is not None:
                    param.grad = None
                else: print(f'none grad on {update_for} : generator')

        elif update_for == 'generator':

            for param in discriminator:
                if param.grad is not None:
                    param.grad = None
                else: print(f'none grad on {update_for} : discriminator')

            for param in generator:
                if param.grad is not None:
                    if maximize_loss:
                        param += lr * param.grad / batch_size
                    else:
                        param -= lr * param.grad / batch_size
                    param.grad = None
                else: print(f'none grad on {update_for} : generator')



def interact(generator): return generator.forward()


# helpers


def required_kernel_size(this_size, target_size): return this_size - (target_size-1) * stride
