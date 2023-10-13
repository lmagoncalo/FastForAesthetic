import clip
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
# from pympler.tracker import SummaryTracker
### stable diffusion
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_input_mob
from keras.layers import Dense, Dropout
# NIMA classifier imports
from keras.models import Model
from scipy.interpolate import interp1d
from torch.nn import functional as F
from torchvision import transforms
# from datasets import load_dataset
from torchvision.transforms import functional as TF
from deap import base
from deap import cma
from deap import creator
from deap import tools

from clip_fitness_cmaes import ClipPrompt
from config import *
from fast_pixel_cmaes import FastPixelRenderer


#### brisque
def init_brisque():
    brisque_model_path = "models/brisque_model_live.yml"
    brisque_range_path = "models/brisque_range_live.yml"
    brisque_model = cv2.quality.QualityBRISQUE_create(brisque_model_path, brisque_range_path)
    return brisque_model


#### nima aesthetic
def init_aesthetics():
    return init_nima("models/weights_mobilenet_aesthetic_0.07.hdf5")


#### nima technical
def init_technical():
    return init_nima("models/weights_mobilenet_technical_0.11.hdf5")


# calculate mean score for AVA dataset
def mean_score(scores):
    si = np.arange(1, 11, 1)
    mean = np.sum(scores * si)
    return mean


# calculate standard deviation of scores for AVA dataset
def std_score(scores):
    si = np.arange(1, 11, 1)
    mean = mean_score(scores)
    std = np.sqrt(np.sum(((si - mean) ** 2) * scores))
    return std


def init_nima(weights_file):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.load_weights(weights_file)
    return model


### simulacra
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


def eval_simulacra(image_numpy, model, clip_model, normalizer):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = Image.fromarray((image_numpy * 1).astype(np.uint8))
    img = TF.resize(img, 224, transforms.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (224, 224))
    img = TF.to_tensor(img).to(device)
    img = normalizer(img)
    clip_image_embed = F.normalize(clip_model.encode_image(img[None, ...]).float(), dim=-1)
    score = model(clip_image_embed)
    # return score.item()
    return score


#### laion
def init_diffusion():
    model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
    s = torch.load(
        "models/sac+logos+ava1-l14-linearMSE.pth", map_location=torch.device('cpu'))  # load the model you trained previously or the model available in this repo
    model.load_state_dict(s)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64

    return model, model2, preprocess


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


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def eval_diffusion(image_numpy, model, model2, preprocess):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # pil_image = Image.open(img_path)
    pil_image = Image.fromarray((image_numpy * 1).astype(np.uint8))
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model2.encode_image(image)

    im_emb_arr = normalized(image_features.cpu().detach().numpy())

    prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.FloatTensor))

    return prediction.item()


#### evaluations ####
def eval_brisque(image_numpy, model=None):
    if model is None:
        model = brisque_model
    score = model.compute(image_numpy)
    # with the score in the first element. The score ranges from 0 (best quality) to 100 (worst quality)
    # print(score)
    new_score = 100 - score[0]
    m = interp1d([0, 100], [1, 10])
    new_score = m(new_score)
    # print(new_score)
    return new_score


def eval_nima(images_numpy, model):
    # NIMA classifier (tensorgp)
    with tf.device('/CPU:0'):
        x = np.stack([images_numpy[index] for index in range(len(images_numpy))], axis=0)
        x = preprocess_input_mob(x)
        scores = model.predict(x, batch_size=len(images_numpy), verbose=0)
        # print(scores)
        final_scores = []
        for index in range(len(images_numpy)):
            mean = mean_score(scores[index])
            std = std_score(scores[index])
            final_scores += [mean]

    return final_scores


def inits():
    brisque_model = init_brisque()
    nima_tec = init_technical()
    nima_aes = init_aesthetics()
    diff_model, diff_model_2, preprocess = init_diffusion()
    simu_model, simu_clip, simu_norm = init_simulacra()
    return brisque_model, nima_tec, nima_aes, diff_model, diff_model_2, preprocess, simu_model, simu_clip, simu_norm



def evaluate(individual):
    np_image = renderer.render(individual)

    brisque_val = eval_brisque(np_image, brisque_model)
    nima_tec_val = eval_nima([np_image], nima_tec)[0]
    nima_aes_val = eval_nima([np_image], nima_aes)[0]
    laion_val = eval_diffusion(np_image, diff_model, diff_model_2, preprocess)
    simulacra_val = eval_simulacra(np_image, simu_model, simu_clip, simu_norm)

    # print("Brisque:", brisque_val, "NIMA tec:", nima_tec_val, "NIMA aes:", nima_aes_val,
    # "Laion:", laion_val, "Simulacra:", simulacra_val.item())

    return brisque_val + nima_tec_val + nima_aes_val + laion_val + simulacra_val.item()


if __name__ == "__main__":
    renderer = FastPixelRenderer()

    print("init all")
    brisque_model, nima_tec, nima_aes, diff_model, diff_model_2, preprocess, simu_model, simu_clip, simu_norm = inits()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", evaluate)
    strategy = cma.Strategy(centroid=renderer.generate_individual(), sigma=SIGMA, lambda_=POP_SIZE)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    for cur_iteration in range(ITERATIONS):
        print("Iteration", cur_iteration)

        # Generate a new population
        population = toolbox.generate()
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = [fit]

        if SAVE_ALL:
            for index, ind in enumerate(population):
                img = renderer.render(ind)
                img = Image.fromarray(img)
                img.save(f"images/resultados_{cur_iteration}_{index}.png")

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        # Update the hall of fame and the statistics with the
        # currently evaluated population
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=cur_iteration, **record)

        if halloffame is not None:
            print("Best individual:", halloffame[0].fitness.values)
            img = renderer.render(halloffame[0])
            img = Image.fromarray(img)
            img.save(f"images/resultados_{cur_iteration}_best.png")

        print(logbook)

