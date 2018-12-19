import Models
import res

import random


    # models

generator = (205, 155)

discriminator = ((110,110),(45,45))


    # params


noise_size = 256

hm_filters1 = 4

hm_filters2 = 5

width = 128
height = 128


hm_epochs  = 20
hm_data    = 40
batches_of = 20

gen_maximize_loss = False
learning_rate     = 0.001


    #


generator = Models.Generator(noise_size, hm_filters1, hm_filters2, width, height, generator)
discriminator = Models.Discriminator(width, height, hm_filters2, hm_filters1, discriminator)


    #


for i in range(hm_epochs):

    epoch_loss_gen, epoch_loss_disc = 0, 0

    for j in range(int(hm_data/batches_of)):

        # fake_data = generator.forward(batchsize=batches_of)       # todo : unlock when data comes in .
        # real_data = res.get_data(hm_data)
        #
        # fake_set = tuple([(data, 0) for data in fake_data])
        # real_set = tuple([(data, 1) for data in real_data])
        #
        # dataset = tuple(random.shuffle([e for e in fake_set + real_set]))
        #
        #

        # for sample, label in dataset: # todo no sample label!!!
        #
        #     discriminator_result = discriminator.forward(sample)
        #     loss = Models.loss_discriminator(discriminator_result, label)
        #
        #     Models.update(loss, discriminator, generator, update_for='discriminator', lr=learning_rate, batch_size=batches_of)
        #
        #     epoch_loss_disc += float(loss)
        # print(f'Epoch {i} Loss Disc: {epoch_loss_disc}')



        constructed_data = generator.forward(batchsize=batches_of)
        discriminator_result = discriminator.forward(constructed_data)

        if gen_maximize_loss:
            loss = Models.loss_generator(discriminator_result, loss_type='maximize')
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', maximize_loss=True, lr=learning_rate, batch_size=batches_of)

        else:
            loss = Models.loss_generator(discriminator_result)
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', lr=learning_rate, batch_size=batches_of)

        print(f'Epoch {i} Loss Gen: {epoch_loss_gen}')


res.pickle_save(discriminator, 'discriminator.pkl')
res.pickle_save(generator, 'generator.pkl')

print('Training is complete.')







# generator = Models.Generator(noise_size, hm_filters1, hm_filters2, width, height)
# discriminator = Models.Discriminator(width, height, hm_filters2, hm_filters1)
#
# generator_out = generator.forward(batchsize=batches_of)
# discriminator_out = discriminator.forward(generator_out)
# print(discriminator_out)
