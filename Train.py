import Models


# params

noise_size = 256

hm_filters1 = 4

hm_filters2 = 5

width = 128
height = 128


generator = Models.Generator(noise_size, hm_filters1, hm_filters2, width, height)
discriminator = Models.Discriminator(width, height, hm_filters2, hm_filters1)


generator_out = generator.forward()
discriminator_out = discriminator.forward(generator_out)
print(discriminator_out)
