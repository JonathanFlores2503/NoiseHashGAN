# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 01:31:08 2023

@author: jonat
"""
import cv2
import yaml 
import torch
import numpy as np
from CLIP import clip
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from taming.models.vqgan import VQModel

class Parameters(torch.nn.Module):
    def __init__(self, noiseHash):
        super(Parameters, self).__init__()
        self.data = 0.5 * noiseHash.view(batch_size, 256,  size1//16, size2//16).cuda()
        self.data = torch.nn.Parameter(torch.sin(self.data))
    def forward(self):
        return self.data

def init_params(noiseHash):
    params = Parameters(noiseHash).cuda()
    optimizer = torch.optim.AdamW([{'params': [params.data], 'lr': learning_rate}], weight_decay=wd)
    return params, optimizer

def show_from_tensor(tensor, pathDirGeneral, nameNoise_Hash):
    img = tensor.clone()
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(pathDirGeneral + "/" + nameNoise_Hash[:-14] + "_NoiseZ"  + ".png", bbox_inches='tight', pad_inches=0)
    plt.show()

def norm_data(data):
    return (data.clip(-1, 1) + 1) / 2  # Range between 0 and 1 in the result

def load_config(config_path, display=False):
    config_data = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config_data)))
    return config_data

def load_vqgan(config, chk_path=None):
    model = VQModel(**config.model.params)
    if chk_path is not None:
        state_dict = torch.load(chk_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model.eval()

def generator(x):
    x = taming_model.post_quant_conv(x)
    x = taming_model.decoder(x)
    return x

def hashToNoiseZ(pathDirGeneral, nameNoise_Hash):
    torch.cuda.empty_cache()
    imageHash = cv2.imread(pathDirGeneral + nameNoise_Hash)
    size1, size2, _ = imageHash.shape
    imageHash = cv2.resize(imageHash, (size1//16,size2//16),interpolation=cv2.INTER_AREA)
    image_gray = cv2.cvtColor(imageHash, cv2.COLOR_BGR2GRAY)
    imageGray_norm = (image_gray.astype(np.float32) / 255.0 - 0.5) * 2.0
    imageGray_pross = np.repeat(imageGray_norm[np.newaxis, :, :], 256, axis=0)
    noiseHash = torch.tensor(imageGray_pross).unsqueeze(0)
    return noiseHash, size1, size2

device = torch.device("cuda:0")
taming_config = load_config("../NoiseHashGAN/confg/model.yaml", display=True)
# !wget 'https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1' -O 'models/vqgan_imagenet_f16_16384/checkpoints/last.ckpt' 
# !wget 'https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1' -O 'models/vqgan_imagenet_f16_16384/configs/model.yaml' 
taming_model = load_vqgan(taming_config, chk_path="../NoiseHashGAN/checkpoints/last.ckpt").to(device)
pathDirGeneral = "../NoiseHashGAN/TestResult/Person_1/"
nameNoise_Hash = "person_1_hashNoise.jpg"  # Changes
wd = 0.1
batch_size = 1
learning_rate = 0.5
total_iter = 400
noise_factor = 0.22
clipmodel, _ = clip.load('ViT-B/32', jit=False)
clipmodel.eval()
print(clip.available_models())
print("Clip model visual input resolution: ", clipmodel.visual.input_resolution)
noiseHash, size1, size2 = hashToNoiseZ(pathDirGeneral, nameNoise_Hash)
Params, optimizer = init_params(noiseHash)
with torch.no_grad():
    print(Params().shape)
    img = norm_data(generator(Params()).cpu())
    print("img dimensions: ", img.shape)
    show_from_tensor(img[0], pathDirGeneral, nameNoise_Hash)



