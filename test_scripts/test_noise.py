import matplotlib.pyplot as plt
import numpy as np
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)

np.random.seed(0)
noise = generate_perlin_noise_2d((1000, 1000), (10, 10))
noise[noise < 0.5 ] = 0
noise[noise >= 0.5 ] = 1

plt.imshow(noise, cmap='gray', interpolation='lanczos')
plt.colorbar()


plt.show()