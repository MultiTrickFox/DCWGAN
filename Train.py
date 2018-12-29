from random import shuffle

from torch import stack

import Models

import res


    # models


generator = (16, 32, 64)

discriminator = (2, )


g_filters = (6, 8, 9)
d_filters = (2, )


    # params


noise_size = 10

width = 256
height = 256


hm_sessions = 1
hm_epochs   = 20
hm_data     = 100
batches_of  = 20

gen_maximize_loss = False
gen_learning_rate     = 0.0005
disc_learning_rate    = 0.0005


    #

data = res.get_data(hm_data)

generator = Models.Generator(noise_size, generator, g_filters, width, height)# ; generator = res.pickle_load('generator.pkl')
discriminator = Models.Discriminator(width, height, discriminator, d_filters)# ; discriminator = res.pickle_load('discriminator.pkl')

# print(cuda.memory_allocated())
# print(cuda.memory_cached())

    #


losses = ([], [])

for j in range(hm_sessions):

    for i in range(hm_epochs):

        shuffle(data) ; data_batches = res.batchify(data, batches_of)

        epoch_loss_gen, epoch_loss_disc = 0, 0

        for real_data in data_batches:


            fake_data = generator.forward(batchsize=batches_of)
            real_data = stack(real_data, 0).to('cuda')

            disc_result_fake = discriminator.forward(fake_data)
            disc_result_real = discriminator.forward(real_data)

            loss = Models.loss_discriminator_w(disc_result_real, disc_result_fake)

            epoch_loss_disc += float(loss)
            Models.update(loss, discriminator, generator, update_for='discriminator', lr=disc_learning_rate, batch_size=batches_of)

            fake_data = generator.forward(batchsize=batches_of)

            disc_result_fake = discriminator.forward(fake_data)

            loss = Models.loss_generator(disc_result_fake) if not gen_maximize_loss \
                else Models.loss_generator(disc_result_fake, loss_type='maximize')

            epoch_loss_gen += float(loss)
            Models.update(loss, discriminator, generator, update_for='generator', lr=gen_learning_rate, batch_size=batches_of)


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
