import torch
from torch.nn import functional as F

from config import *


class FastPixelRenderer:
    def __init__(self):
        super(FastPixelRenderer, self)

        self.individual = torch.rand(1, 3, NUM_PIXELS, NUM_PIXELS).float().to(DEVICE)
        self.individual.requires_grad = True

        self.optimizer = None

    def get_optim(self):
        self.optimizer = torch.optim.Adam([self.individual], lr=LR)

        return [self.optimizer]

    def __str__(self):
        return "fastpixeldraw"

    def render(self):
        img = F.interpolate(self.individual, size=(IMG_SIZE, IMG_SIZE), mode="nearest")
        return img
