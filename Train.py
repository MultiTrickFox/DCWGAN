import Models
import res

from random import shuffle
from torch import stack, Tensor, tensor



    # models


generator = (150, )

discriminator = (25, )


g_filters = (2, )
d_filters = (2, )


    # params


noise_size = 50

width = 256
height = 256


hm_epochs  = 50
hm_data    = 324
batches_of = 9

gen_maximize_loss = False
learning_rate     = 0.0002


    #


generator = Models.Generator(noise_size, generator, g_filters, width, height)# ; generator = res.pickle_load('generator.pkl')
discriminator = Models.Discriminator(width, height, discriminator, d_filters)# ; discriminator = res.pickle_load('discriminator.pkl')


    #


losses = ([], [])

for i in range(hm_epochs):

    epoch_loss_gen, epoch_loss_disc = 0, 0

    batches = res.batchify(res.get_data(hm_data), batches_of)

    for real_data in batches:

        fake_data = generator.forward(batchsize=batches_of)

        real_set = tuple([(data, tensor(1)) for data in real_data])
        fake_set = tuple([(data, tensor(0)) for data in fake_data])

        all_set = [e for e in fake_set + real_set]
        shuffle(all_set)

        databox = []
        labelbox = []
        for e in all_set:
            databox.append(e[0])
            labelbox.append(e[1])

        discriminator_result = discriminator.forward(stack(databox, 0))
        loss = Models.loss_discriminator(discriminator_result, Tensor(labelbox))

        Models.update(loss, discriminator, generator, update_for='discriminator', lr=learning_rate, batch_size=batches_of)

        epoch_loss_disc += float(loss)

        fake_data = generator.forward(batchsize=batches_of)
        discriminator_result = discriminator.forward(fake_data)

        if gen_maximize_loss:
            loss = Models.loss_generator(discriminator_result, loss_type='maximize')
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', maximize_loss=True, lr=learning_rate, batch_size=batches_of)

        else:
            loss = Models.loss_generator(discriminator_result)
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
