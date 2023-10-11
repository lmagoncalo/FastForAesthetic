import subprocess
from pathlib import Path

import clip
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from config import DEVICE


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)


class AestheticLoss:
    def __init__(self, model=None, preprocess=None, clip_model="ViT-B/16"):
        super(AestheticLoss, self).__init__()

        self.aesthetic_target = 1
        # Only available here: https://twitter.com/RiversHaveWings/status/1472346186728173568
        self.model_path = Path("models/ava_vit_b_16_linear.pth")

        if not self.model_path.exists():
            wget_file(
                "https://cdn.discordapp.com/attachments/821173872111517696/921905064333967420/ava_vit_b_16_linear.pth",
                self.model_path)

        layer_weights = torch.load(self.model_path)
        self.ae_reg = nn.Linear(512, 1).to(DEVICE)
        self.ae_reg.load_state_dict(torch.load(self.model_path))
        # self.ae_reg.bias.data = layer_weights["bias"].to(self.device)
        # self.ae_reg.weight.data = layer_weights["weight"].to(self.device)

        self.clip_model = "ViT-B/16"

        if model is None:
            print(f"Loading CLIP model: {clip_model}")

            self.model, self.preprocess = clip.load(self.clip_model, device=DEVICE)

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
        aes_rating = self.ae_reg(F.normalize(image_features.float(), dim=-1))

        return aes_rating
