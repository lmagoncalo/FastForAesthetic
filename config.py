import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224

NUM_PIXELS = 28
NUM_LINES = 10

LR = 0.1

ITERATIONS = 500
