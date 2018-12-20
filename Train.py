import Models
import res

from random import shuffle
from torch import stack, Tensor



    # models


generator = (320, 280)

discriminator = (128, 64)


    # params


noise_size = 400

hm_filters1 = 4

hm_filters2 = 5

width = 256
height = 256


hm_epochs  = 20
hm_data    = 2# 300
batches_of = 1# 5

gen_maximize_loss = False
learning_rate     = 0.001


    #


generator = Models.Generator(noise_size, hm_filters1, hm_filters2, width, height, generator)
discriminator = Models.Discriminator(width, height, hm_filters2, hm_filters1, discriminator)


    #


losses = ([], [])

for i in range(hm_epochs):

    epoch_loss_gen, epoch_loss_disc = 0, 0

    batches = res.batchify(res.get_data(hm_data), batches_of)

    for real_data in batches:

        fake_data = generator.forward(batchsize=batches_of)

        real_set = tuple([(data, 1) for data in real_data])
        fake_set = tuple([(data, 0) for data in fake_data])

        all_set = [e for e in fake_set + real_set]
        shuffle(all_set)

        databox = []
        labelbox = []
        for e in all_set:
            databox.append(e[0])
            labelbox.append(e[1])

        discriminator_result = discriminator.forward(stack(databox, 0))
        loss = Models.loss_discriminator(discriminator_result.to('cpu'), Tensor(labelbox).to('cpu'))

        Models.update(loss, discriminator, generator, update_for='discriminator', lr=learning_rate, batch_size=batches_of)

        epoch_loss_disc += float(loss)

        constructed_data = generator.forward(batchsize=batches_of)
        discriminator_result = discriminator.forward(constructed_data)

        if gen_maximize_loss:
            loss = Models.loss_generator(discriminator_result.to('cpu'), loss_type='maximize')
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', maximize_loss=True, lr=learning_rate, batch_size=batches_of)

        else:
            loss = Models.loss_generator(discriminator_result.to('cpu'))
            epoch_loss_gen += float(loss)

            Models.update(loss, discriminator, generator, update_for='generator', lr=learning_rate, batch_size=batches_of)


    print(f'Epoch {i} Loss Disc : {round(epoch_loss_disc,3)} Loss Gen : {round(epoch_loss_gen,3)}')
    for e,ee in zip(losses, (epoch_loss_disc, epoch_loss_gen)): e.append(ee)

res.pickle_save(discriminator, 'discriminator.pkl')
res.pickle_save(generator, 'generator.pkl')

print('Training is complete.')





import matplotlib.pyplot as plot

for _, color in enumerate(('g', 'b')):
    plot.plot(range(hm_epochs), losses[_], color)
plot.show()


import torchvision.transforms.functional as F

for _ in range(5):

    result = generator.forward().view(3, width, height)
    result_arr = result.detach().cpu()
    F.to_pil_image(result_arr).save("result"+str(_+1)+".jpg")

