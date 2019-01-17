from random import shuffle, choice
from Models import RMS, Adam
from torch import stack

import Models
import res


    # models


gen_struct = (4, 8, 16, 32, 64, 128)
g_filters  = (256, 128, 64, 32, 16, 8)

disc_struct = (128, 64, 32, 16, 8, 4)
d_filters   = (8, 16, 32, 64, 16, 4)


    # params


noise_size = 100

width = 128
height = 128


hm_sessions = 2
hm_epochs   = 20
hm_data     = 200
batches_of  = 10

gen_maximize_loss  = False
gen_learning_rate  = 1e-4
disc_learning_rate = 1e-4

hm = 1


    #


new_discriminator = True
new_generator     = True


generator = res.pickle_load('generator.pkl')
discriminator = res.pickle_load('discriminator.pkl')
if new_generator or generator is None:
    generator = Models.Generator(noise_size, gen_struct, g_filters, width, height)
if new_discriminator or discriminator is None:
    discriminator = Models.Discriminator(width, height, disc_struct, d_filters)

optimizers = (RMS(discriminator.parameters(), disc_learning_rate), Adam(generator.parameters(), gen_learning_rate))


    #


losses = ([], [])

for j in range(hm_sessions):

    data = res.get_data(hm_data)

    for i in range(hm_epochs):

        shuffle(data) ; data_batches = res.batchify(data, batches_of)

        epoch_loss_gen, epoch_loss_disc = 0, 0

        for real_data in data_batches:


            fake_data = generator.forward(batchsize=batches_of)
            real_data = stack(real_data, 0).to('cuda')

            disc_result_fake = discriminator.forward(fake_data)
            disc_result_real = discriminator.forward(real_data)

            loss = Models.loss_discriminator(disc_result_real, 1) + Models.loss_discriminator(disc_result_fake, 1)
            # loss = Models.loss_discriminator_w(disc_result_real, disc_result_fake)

            epoch_loss_disc += float(loss)/hm
            Models.update(loss, discriminator, generator, update_for='discriminator', optimizers=optimizers, batch_size=batches_of)

            for _ in range(hm-1):
                fake_data = generator.forward(batchsize=batches_of)
                real_data = stack(choice(data_batches), 0).to('cuda')

                loss = Models.loss_discriminator_w(disc_result_real, disc_result_fake)

                epoch_loss_disc += float(loss)/hm
                Models.update(loss, discriminator, generator, update_for='discriminator', optimizers=optimizers, batch_size=batches_of)


            fake_data = generator.forward(batchsize=batches_of)

            disc_result_fake = discriminator.forward(fake_data)

            loss = Models.loss_generator(disc_result_fake)

            epoch_loss_gen += float(loss) # / hm
            Models.update(loss, discriminator, generator, update_for='generator', optimizers=optimizers, batch_size=batches_of)

            # for _ in range(hm - 1):
            #     fake_data = generator.forward(batchsize=batches_of)
            #
            #     disc_result_fake = discriminator.forward(fake_data)
            #
            #     loss = Models.loss_generator(disc_result_fake, type='maximize' if gen_maximize_loss else 'minimize')
            #
            #     epoch_loss_gen += float(loss) / hm
            #     Models.update(loss, discriminator, generator, update_for='generator', lr=disc_learning_rate, batch_size=batches_of)


            print('/', end='', flush=True)
        print(f'\n {res.get_clock()} s{j+1}e{i+1} - Loss Disc : {round(epoch_loss_disc,3)} , Loss Gen : {round(epoch_loss_gen,3)}')
        losses[0].append(epoch_loss_gen) ; losses[1].append(epoch_loss_disc)
    res.imgmake(generator, hm=5)
print('Training is complete.')


res.plot(losses)


if input('hit n to NOT SAVE..: ') != 'n':
    res.pickle_save(discriminator, 'discriminator.pkl')
    res.pickle_save(generator, 'generator.pkl')
    print('work saved.')
