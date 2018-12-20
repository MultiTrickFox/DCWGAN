import torch
from torch import nn

from multiprocessing.pool import ThreadPool as Pool
from torch.multiprocessing import cpu_count


# global declarations

hm_channels = 3
stride      = 1

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# models


class Generator(nn.Module):
    def __init__(self, noise_size, hm_filters1, hm_filters2, width_out, height_out, layers):
        super(Generator, self).__init__()
        self.noise_size = noise_size

        self.model = nn.Sequential(

            nn.Conv2d(in_channels  = hm_channels,
                      out_channels = hm_filters1,
                      kernel_size  = (required_kernel_size(noise_size, layers[0]),
                                      required_kernel_size(noise_size, layers[0])),
                      stride       = stride,
                      bias         = False),
            nn.BatchNorm2d(hm_filters1),
            nn.ReLU(),

            nn.Conv2d(in_channels  = hm_filters1,
                      out_channels = hm_filters2,
                      kernel_size  = (required_kernel_size(int(layers[0]), int(layers[1])),
                                      required_kernel_size(int(layers[0]), int(layers[1]))),
                      stride       = stride,
                      bias         = False),
            nn.BatchNorm2d(hm_filters2),
            nn.ReLU(),

            nn.Conv2d(in_channels  = hm_filters2,
                      out_channels = hm_channels,
                      kernel_size  = (required_kernel_size(int(layers[1]), width_out),
                                      required_kernel_size(int(layers[1]), height_out)),
                      stride       = stride,
                      bias         = False),
            nn.BatchNorm2d(hm_channels),
            nn.Tanh(),
        )

    def forward(self, batchsize=1): return self.model(torch.randn(batchsize, hm_channels, self.noise_size, self.noise_size).to('cuda'))


class Discriminator(nn.Module):
    def __init__(self, width_in, height_in, hm_filters1, hm_filters2, layers):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(in_channels  = hm_channels,
                      out_channels = hm_filters1,
                      kernel_size  = (required_kernel_size(width_in, layers[0]),
                                      required_kernel_size(height_in, layers[0])),
                      stride       = stride,
                      bias         = False),
            nn.BatchNorm2d(hm_filters1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels  = hm_filters1,
                      out_channels = hm_filters2,
                      kernel_size  = (required_kernel_size(layers[0], layers[1]),
                                      required_kernel_size(layers[0], layers[1])),
                      stride       = stride,
                      bias         = False),
            nn.BatchNorm2d(hm_filters2),
            nn.LeakyReLU(),
        )

        self.model2 = nn.Sequential(

            nn.Linear(layers[1] * layers[1] * hm_filters2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inp):
        result_pre = self.model(inp)
        result = self.model2(result_pre.view(result_pre.size(0), -1))
        return result


def loss_discriminator(discriminator_results, labels):
    with Pool(cpu_count()) as p:
        results = p.map(fn_disc, tuple([(discriminator_result, label) for discriminator_result, label in zip(discriminator_results, labels)]))

        p.close()
        p.join()

    return sum(results)

def fn_disc(data):
    output, target = data
    if target == 1:
        return (- torch.log(output)).sum()
    else:
        return (- torch.log(1 - output)).sum()


def loss_generator(discriminator_results, loss_type='minimize'):
    with Pool(cpu_count()) as p:
        results = p.map(fn_disc, tuple([(discriminator_result, loss_type) for discriminator_result in discriminator_results]))

        p.close()
        p.join()

    return sum(results)


def fn_gen(data):
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
