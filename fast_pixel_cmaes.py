import cv2
import numpy as np

from config import *


class FastPixelRenderer:
    def __init__(self):
        super(FastPixelRenderer, self)

    def generate_individual(self):
        individual = np.random.rand(NUM_PIXELS, NUM_PIXELS, 3)
        return individual.flatten()

    def __str__(self):
        return "fastpixeldraw"

    def render(self, individual):
        ind = np.reshape(individual, (NUM_PIXELS, NUM_PIXELS, 3))
        # img = np.resize(ind, (224, 224, 3))
        img = cv2.resize(ind, dsize=(224, 224), interpolation=cv2.INTER_NEAREST)
        img *= 255
        img = img.astype(np.uint8)
        return img
