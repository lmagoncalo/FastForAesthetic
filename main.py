import torch
from torchvision.utils import save_image
from tqdm import tqdm

from aesthetic_loss import AestheticLoss
from clip_loss import ClipPrompt
from fast_pixel import FastPixelRenderer

from config import ITERATIONS

loss_function = AestheticLoss()
# loss_function = ClipPrompt(prompt="A panda")

render = FastPixelRenderer()

optimizers = render.get_optim()

for i in tqdm(range(ITERATIONS + 1)):
    # May have multiple optimizers, se we need to iterate each one
    for optimizer in optimizers:
        optimizer.zero_grad()

    # Render a new image
    img = render.render()

    # Evaluate the image with the aesthetic model
    loss = loss_function.evaluate(img)

    # Backpropagate
    loss.backward()

    for optimizer in optimizers:
        optimizer.step()

    # Save the image each 20 iterations
    if i % 20 == 0:
        print("Loss: ", loss.item())
        save_image(img, f"images/img_{i}.png")

