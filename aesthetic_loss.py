import subprocess
from pathlib import Path

import clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from config import DEVICE, CLIP_MODEL, AESTHETIC_MODEL


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class AestheticLoss:
    def __init__(self, model=None, preprocess=None):
        super(AestheticLoss, self).__init__()

        self.aesthetic_target = 1
        # Only available here: https://twitter.com/RiversHaveWings/status/1472346186728173568
        self.model_path = Path(f"models/{AESTHETIC_MODEL}")

        if CLIP_MODEL == "ViT-L/14":
            self.ae_reg = nn.Linear(768, 1).to(DEVICE)
        elif CLIP_MODEL == "ViT-B/32" or CLIP_MODEL == "ViT-B/16":
            self.ae_reg = nn.Linear(512, 1).to(DEVICE)
        else:
            raise ValueError()

        self.ae_reg.load_state_dict(torch.load(self.model_path))

        if model is None:
            print(f"Loading CLIP model: {CLIP_MODEL}")

            self.model, self.preprocess = clip.load(CLIP_MODEL, device=DEVICE)

            print("CLIP module loaded.")
        else:
            self.model = model
            self.preprocess = preprocess

    def evaluate(self, img):
        img = img.to(DEVICE)

        img = TF.resize(img, [224, 224])

        image_features = self.model.encode_image(img)

        # Here are the weights:
        # https://cdn.discordapp.com/attachments/821173872111517696/921905064333967420/ava_vit_b_16_linear.pth,
        # you load them into an nn.Linear(512, 1) and then pre-normalize the ViT-B/16 embeddings you feed into it so
        # that they have norm 1 (i.e. model(F.normalize(embeds, dim=-1))).
        aes_loss = self.ae_reg(F.normalize(image_features.float(), dim=-1))
        aes_loss = (aes_loss - 10).square().mean() * 0.02

        return aes_loss
