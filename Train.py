from random import shuffle

from torch import cuda

import Models

import res


    # models


generator = (50, 150)

discriminator = (50, 10)


g_filters = (5, 4)
d_filters = (5, 2)


    # params


noise_size = 10

width = 256
height = 256


hm_epochs  = 50
hm_data    = 324
batches_of = 9

gen_maximize_loss = False
learning_rate     = 0.0005


    #

data = res.get_data(hm_data)

generator = Models.Generator(noise_size, generator, g_filters, width, height)# ; generator = res.pickle_load('generator.pkl')
discriminator = Models.Discriminator(width, height, discriminator, d_filters)# ; discriminator = res.pickle_load('discriminator.pkl')

print(cuda.memory_allocated())
print(cuda.memory_cached())

    #


losses = ([], [])

for i in range(hm_epochs):

    epoch_loss_gen, epoch_loss_disc = 0, 0

    shuffle(data)
    data_batches = res.batchify(data, batches_of)

    for real_data in data_batches:

        fake_data = generator.forward(batchsize=batches_of)
        real_data = real_data.to('cuda')

        disc_result_fake = discriminator.forward(fake_data)
        disc_result_real = discriminator.forward(real_data)

        loss = Models.loss_discriminator_w(disc_result_real, disc_result_fake)

        Models.update(loss, discriminator, generator, update_for='discriminator', lr=learning_rate, batch_size=batches_of)

        epoch_loss_disc += float(loss)

        fake_data = generator.forward(batchsize=batches_of)
        disc_result_fake = discriminator.forward(fake_data)

        if gen_maximize_loss:
            loss = Models.loss_generator(disc_result_fake, loss_type='maximize')
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', maximize_loss=True, lr=learning_rate, batch_size=batches_of)

        else:
            loss = Models.loss_generator(disc_result_fake)
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', lr=learning_rate, batch_size=batches_of)

        print('/', end='', flush=True)
    print(f'\n {res.get_clock()} Epoch {i+1} Loss Disc : {round(epoch_loss_disc,3)} Loss Gen : {round(epoch_loss_gen,3)}')
    losses[0].append(epoch_loss_gen) ; losses[1].append(epoch_loss_disc)
print('Training is complete.')





res.plot(losses, hm_epochs)
res.imgmake(generator, hm=5)


if input('hit n to NOT SAVE..: ') != 'n':
    res.pickle_save(discriminator, 'discriminator.pkl')
    res.pickle_save(generator, 'generator.pkl')
    print('work saved.')
