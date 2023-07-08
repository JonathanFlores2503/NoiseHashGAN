#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 21:17:09 2023

@author: pc
"""
import cv2
import yaml 
import torch
import numpy as np
from CLIP import clip
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from taming.models.vqgan import VQModel
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class Parameters(torch.nn.Module):
    def __init__(self):
        super(Parameters, self).__init__()
        self.data = 0.5 * torch.full((batch_size, 256, size1//16, size2//16), 0.5).cuda()
        # self.data = .5*torch.randn(batch_size, 256, size1//16, size2//16).cuda()
        self.data = torch.nn.Parameter(torch.sin(self.data))
    def forward(self):
        return self.data

def init_params():
    params = Parameters().cuda()
    optimizer = torch.optim.AdamW([{'params': [params.data], 'lr': learning_rate}], weight_decay=wd)
    return params, optimizer

def show_from_tensor(tensor, pathDirGeneral, numeGenerate, save):
    img = tensor.clone()
    img = img.mul(255).byte()
    img = img.cpu().numpy().transpose((1, 2, 0))
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.imshow(img)
    if save == 1:
        plt.savefig(pathDirGeneral + "/" + "NoiseZ_GAN_" + numeGenerate + ".png", bbox_inches='tight', pad_inches=0)
    elif save == 2:
        plt.savefig(pathDirGeneral + "/" + "GAN_" + numeGenerate + ".png", bbox_inches='tight', pad_inches=0)
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

def encodeText(text):
  t=clip.tokenize(text).cuda()
  t=clipmodel.encode_text(t).detach().clone()
  return t

def createEncodings(include, exclude, extras):
  include_enc=[]
  for text in include:
    include_enc.append(encodeText(text))
  exclude_enc=encodeText(exclude) if exclude != '' else 0
  extras_enc=encodeText(extras) if extras !='' else 0
  return include_enc, exclude_enc, extras_enc

def create_crops(img, num_crops=32): 
  p=size1//2
  img = torch.nn.functional.pad(img, (p,p,p,p), mode='constant', value=0) 
  img = augTransform(img)
  crop_set = []
  for ch in range(num_crops):
    gap1= int(torch.normal(1.2, .3, ()).clip(.43, 1.9) * size1)
    offsetx = torch.randint(0, int(size1*2-gap1),())
    offsety = torch.randint(0, int(size1*2-gap1),())
    crop=img[:,:,offsetx:offsetx+gap1, offsety:offsety+gap1]
    while crop.size(2) == 0 or crop.size(3) == 0:
        offsetx = torch.randint(0, int(size1*2-gap1), ())
        offsety = torch.randint(0, int(size1*2-gap1), ())
        crop = img[:, :, offsetx:offsetx+gap1, offsety:offsety+gap1]
    crop = torch.nn.functional.interpolate(crop,(224,224), mode='bilinear', align_corners=True)
    crop_set.append(crop)
  img_crops=torch.cat(crop_set,0) 
  randnormal = torch.randn_like(img_crops, requires_grad=False)
  num_rands=0
  randstotal=torch.rand((img_crops.shape[0],1,1,1)).cuda() #32
  for ns in range(num_rands):
    randstotal*=torch.rand((img_crops.shape[0],1,1,1)).cuda()
  img_crops = img_crops + noise_factor*randstotal*randnormal
  return img_crops

def showme(Params, show_crop):
  with torch.no_grad():
    generated = generator(Params())
    if (show_crop):
      print("Augmented cropped example")
      aug_gen = generated.float()
      aug_gen = create_crops(aug_gen, num_crops=1)
      aug_gen_norm = norm_data(aug_gen[0])
      show_from_tensor(aug_gen_norm, pathDirGeneral, numeGenerate, 2)
    print("Generation")
    latest_gen=norm_data(generated.cpu())
    show_from_tensor(latest_gen[0],pathDirGeneral, numeGenerate, 2)
  return (latest_gen[0]) 

def optimize_result(Params, prompt):
  alpha=1 
  beta=.5
  out = generator(Params())
  out = norm_data(out)
  out = create_crops(out)
  out = normalize(out)
  image_enc=clipmodel.encode_image(out)
  final_enc = w1*prompt + w1*extras_enc 
  final_text_include_enc = final_enc / final_enc.norm(dim=-1, keepdim=True)
  final_text_exclude_enc = exclude_enc
  main_loss = torch.cosine_similarity(final_text_include_enc, image_enc, -1)
  penalize_loss = torch.cosine_similarity(final_text_exclude_enc, image_enc, -1)
  final_loss = -alpha*main_loss + beta*penalize_loss
  return final_loss

def optimize(Params, optimizer, prompt):
  loss = optimize_result(Params, prompt).mean()
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return loss

def training_loop(Params, optimizer, show_crop=False):
  res_img=[]
  res_z=[]
  for prompt in include_enc:
    iteration=0
    Params, optimizer = init_params()
    for it in range(total_iter):
      loss = optimize(Params, optimizer, prompt)
      if iteration>=1 and iteration%show_step == 0:
        new_img = showme(Params, show_crop)
        res_img.append(new_img)
        res_z.append(Params())
        print("loss:", loss.item(), "\niteration:",iteration)
      iteration+=1
    torch.cuda.empty_cache()
  return res_img, res_z

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
imageHash = cv2.imread(pathDirGeneral + nameNoise_Hash)
size1, size2, _ = imageHash.shape
Params, optimizer = init_params()
with torch.no_grad():
    print(Params().shape)
    img = norm_data(generator(Params()).cpu())
    print("img dimensions: ", img.shape)
    show_from_tensor(img[0], pathDirGeneral, numeGenerate, 1)

normalize = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
# torch.manual_seed(42) #changes
augTransform = torch.nn.Sequential(
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomAffine(30, (.2, .2), fill=0)  
).cuda()

torch.cuda.empty_cache()
include=['A mountainous landscape at sunset']
exclude='watermark, cropped, confusing, incoherent, cut, blurry'
extras = ""
w1=1
w2=1
noise_factor= .22
total_iter=110
show_step=total_iter-1
include_enc, exclude_enc, extras_enc = createEncodings(include, exclude, extras)
res_img, res_z=training_loop(Params, optimizer, show_crop=False)
    