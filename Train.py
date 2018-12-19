import Models
import res

from torch import Tensor


    # models


generator = (416, 312)

discriminator = ((200,200),(128,128))


    # params


noise_size = 512

hm_filters1 = 4

hm_filters2 = 5

width = 256
height = 256


hm_epochs  = 20
hm_data    = 500
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

        real_data = res.get_data(batches_of)
        fake_data = generator.forward(batchsize=batches_of)

        real_set = tuple([(data, 1) for data in real_data])
        fake_set = tuple([(data, 0) for data in fake_data])

        databox = []
        labelbox = []
        for e in fake_set + real_set:

            databox.append(e[0])
            labelbox.append(e[1])

        data = Tensor(databox)
        label = Tensor(labelbox)

        discriminator_result = discriminator.forward(data)
        loss = Models.loss_discriminator(discriminator_result, label)

        Models.update(loss, discriminator, generator, update_for='discriminator', lr=learning_rate, batch_size=batches_of)

        epoch_loss_disc += float(loss)

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


    print(f'Epoch {i} Loss Disc: {epoch_loss_disc}')
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
