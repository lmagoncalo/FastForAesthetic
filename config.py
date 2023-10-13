import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224  # Render output size

CLIP_MODEL = "ViT-B/16"  # CLIP model to be used in the loss

AESTHETIC_MODEL = "ava_vit_b_16_linear.pth"  # Aesthetic model, needs to be adapted to the CLIP model

NUM_PIXELS = 16  # Number of pixel for each side, i.e. 28x28 pixels renderer as a 224x224 image

LR = 0.1  # Learning rate

SIGMA = 0.05

POP_SIZE = 100

ITERATIONS = 200  # Number of iterations

PROMPT = "A panda"  # A prompt for clip just for testing

SAVE_ALL = False
