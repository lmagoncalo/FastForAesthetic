import clip
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm

from config import *
from fast_pixel_adam import FastPixelRenderer


class AestheticMeanPredictionLinearModel(nn.Module):
    def __init__(self, feats_in):
        super().__init__()
        self.linear = nn.Linear(feats_in, 1)

    def forward(self, input):
        x = F.normalize(input, dim=-1) * input.shape[-1] ** 0.5
        return self.linear(x)


def init_simulacra():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model_name = 'ViT-B/16'
    clip_model = clip.load(clip_model_name, jit=False, device=device)[0]
    clip_model.eval().requires_grad_(False)

    normalizer = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                      std=[0.26862954, 0.26130258, 0.27577711])

    # 512 is embed dimension for ViT-B/16 CLIP
    model = AestheticMeanPredictionLinearModel(512)
    model.load_state_dict(
        torch.load("models/sac_public_2022_06_29_vit_b_16_linear.pth")
    )
    model = model.to(device)
    return model, clip_model, normalizer


def eval_simulacra(img, model, clip_model, normalizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # img = Image.fromarray((image_numpy * 1).astype(np.uint8))
    # img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224, 224))
    # img = TF.to_tensor(img).to(device)
    img = img.to(device)
    # img = normalizer(img)
    clip_image_embed = F.normalize(clip_model.encode_image(img).float(), dim=-1)
    score = model(clip_image_embed)
    # return score.item()
    return score


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)



# if you changed the MLP architecture during training, change it also here:
class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            # nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


def init_diffusion():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load(
        "models/sac+logos+ava1-l14-linearMSE.pth", map_location=torch.device('cpu'))  # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)
    model.eval()
    model.to(device)
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    return model, model2, preprocess


def eval_diffusion(img, model, model2, preprocess):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pil_image = Image.open(img_path)
    # pil_image = Image.fromarray((img * 1).astype(np.uint8))
    # img = preprocess(img).unsqueeze(0).to(device)

    clip_image_embed = F.normalize(model2.encode_image(img).float(), dim=-1)

    # prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.FloatTensor))
    prediction = model(clip_image_embed)

    return prediction


# loss_function = ClipPrompt(prompts=PROMPT)
diff_model, diff_model_2, preprocess = init_diffusion()
simu_model, simu_clip, simu_norm = init_simulacra()


render = FastPixelRenderer()

optimizers = render.get_optim()

for i in tqdm(range(ITERATIONS + 1)):
    # May have multiple optimizers, se we need to iterate each one
    for optimizer in optimizers:
        optimizer.zero_grad()

    # Render a new image
    img = render.render()

    # Evaluate the image with the aesthetic model
    # fitness_value = eval_diffusion(img, diff_model, diff_model_2, preprocess)
    fitness_value = eval_simulacra(img, simu_model, simu_clip, simu_norm)

    # Backpropagate
    loss = 1 / (1 + fitness_value)
    loss.backward()

    for optimizer in optimizers:
        optimizer.step()

    # Save the image each 20 iterations
    if i % 20 == 0:
        print("Loss: ", fitness_value.item())
        save_image(img, f"images_simulacra/img_{i}.png")



"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_diffusion_start = Image.open("images_laion/img_0.png")
img_diffusion_start = TF.to_tensor(img_diffusion_start).unsqueeze(0).to(device)
img_diffusion_end = Image.open("images_laion/img_200.png")
img_diffusion_end = TF.to_tensor(img_diffusion_end).unsqueeze(0).to(device)

img_simulacra_start = Image.open("images_simulacra/img_0.png")
img_simulacra_start = TF.to_tensor(img_simulacra_start).unsqueeze(0).to(device)
img_simulacra_end = Image.open("images_simulacra/img_200.png")
img_simulacra_end = TF.to_tensor(img_simulacra_end).unsqueeze(0).to(device)

metric_diffusion_start = eval_diffusion(img_diffusion_start, diff_model, diff_model_2, preprocess)
metric_diffusion_end = eval_diffusion(img_diffusion_end, diff_model, diff_model_2, preprocess)

metric_simulacra_start = eval_simulacra(img_simulacra_start, simu_model, simu_clip, simu_norm)
metric_simulacra_end = eval_simulacra(img_simulacra_end, simu_model, simu_clip, simu_norm)

print("eval_diffusion_start: ", metric_diffusion_start.item(), "eval_diffusion_end: ", metric_diffusion_end.item())
print("eval_simulacra_start: ", metric_simulacra_start.item(), "eval_simulacra_end: ", metric_simulacra_end.item())
"""
