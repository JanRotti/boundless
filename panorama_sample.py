import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import re
import argparse
from utils.utils import *
from utils.parser import parameter_parser
# Preprocessing
import glob
import random
from PIL import Image, ImageOps

# model dependencies
import torch
import torch.nn as nn
import torchvision
from models import Generator

"""Finished"""
def loadModel(epoch = "0", directory = './checkpoints/'):
    """
    Return Pretrained Generator Model for Sampling
    """
    # Security Checks on input
    if not os.path.exists(directory):
        raise ValueError("ValueError directory not found")
    epoch = str(epoch)
    try:
        int(epoch)
    except ValueError:
        print("Non-numeric epoch specification")

    listG = sorted(glob.glob(directory + "G*.pth"))
    
    if  len(listG) == 0:
        print("[*] No Checkpoints found!")
        return 1
    
    ckp_file = ""
    numbers = [re.findall(r'\d+', path)[-1] for path in listG]
    for i in range(len(numbers)):
        if epoch < numbers[i]:
            ckp_file = listG[i-1]
            break

    if not ckp_file:
        ckp_file = listG[-1]

    G = Generator()
    gState = torch.load(ckp_file, map_location='cpu')
    G.load_state_dict(gState)        
    return G

def sample(G, images, counter):
    """
    Samples a image generation
    """
    maskedImg = Variable(images["lr"])
    realImg = Variable(images["hr"])
    mask = Variable(images["alpha"])
    stacked = Variable(images["clip"])
    G.eval()
    fakeImg = G(stacked)
    gStacked = fakeImg * mask + maskedImg
    generated = denormalize(gStacked)
    return counter+1, generated

def adapt_image(generated, stack, maskRatio, sampleDir):
    """
    Adapts Generated Images to Offset
    """
    # Define Size parameters
    generated = torch.squeeze(generated)
    height, width = generated.size()
    crop = int(maskRatio * width)
    # Retrieve Generation to stack
    cropped = generated[:,-crop:]
    stack.append(cropped)
    # Creating an offset image of generation
    offset = torch.zeros((128,128))
    offset[:,:-crop] = generated[:,crop:]
    # Saving offset image
    path = sampleDir + "offset.png"
    torchvision.utils.save_image(offset, path) 
    return stack

def load_image(filepath,imgSize, maskRatio):
    """
    Loads a single image from filepath in training format
    """
    if os.path.exists(filepath):
        img = Image.open(filepath)
        img = ImageOps.grayscale(img) 
    else:
        raise ValueError("ValueError directory not found")
    width, height = img.size
    if width < 128 or height < 128:
        raise ValueError("ValueError ImgSize not sufficient") 
    img = np.expand_dims(np.asarray(img),0).astype("d")
    img = img / 127.5 - 1.0 
    x = 0 #random.randint(0, height - .imgSize)
    y = random.randint(0, width - imgSize)
      
    # Retrieving Img with ImgSize x ImgSize from Bigger Image
    img = img[:, x:x + imgSize, y:y + imgSize]
    # Create Canvas
    canvas = np.zeros((imgSize, imgSize))  
    edge = int(imgSize * maskRatio)
      
    # Create Empty Mask Values
    canvas[:, -edge:] = 1
    canvas = canvas[np.newaxis, :, :]
      
    # Create ImgSize x ImgSize ones Array for Real
    real = torch.ones((imgSize, imgSize), dtype=torch.float)
      
    # Apply Mask
    masked = img * (1 - canvas)
      
    # Transform to torch Tensors
    canvas = torch.from_numpy(canvas).float()
    realImage = torch.from_numpy(img).float()
    maskedImage = torch.from_numpy(masked).float()
      
    # Clip Output in Channel Dimension
    clip = torch.cat([maskedImage, real[None, :, :], canvas])
      
    # return Images as dictionary object
    return {"lr": torch.unsqueeze(maskedImage,0), "hr": torch.unsqueeze(realImage,0), 'alpha': torch.unsqueeze(canvas,0), 'clip': torch.unsqueeze(clip,0)}


def main():
    args = parameter_parser()
    sampleEpoch = args.sampleEpoch
    runName = args.runName
    maskRatio = args.maskRatio
    num = 3 #args.num # number of extensions
    counter = 0
    sampleFile = args.sampleFile
    sampleFile = "/home/jan/Documents/PythonScripts/boundless/data/GPR/B12/B12 Pictures LS/B6N_LGRB_I_R06T_part_00000000_00065535_15.png"
    directory = './checkpoints/' + runName + "/" 
    sampleDir = './samples/' + runName + "/"
    offsetPath = sampleDir + "offset.png"
    G = loadModel(sampleEpoch, directory)
    stack = []
    if os.path.exists(offsetPath):
        os.remove(offsetPath)
    # Performing image extension
    readFile = sampleFile
    while counter < num:
        img = load_image(readFile, 128, maskRatio)
        counter, generated = sample(G, img, counter)
        stack = adapt_image(generated, stack, maskRatio, sampleDir)
        readFile = offsetPath
    # Patching panorama
    patchSize = int(128 * maskRatio)
    width = int(128 + patchSize * (num - 1))
    height = 128
    panorama = torch.zeros((height,width))
    real = denormalize(load_image(sampleFile,128,maskRatio)["hr"])
    panorama[:,:128] = torch.squeeze(real)
    for i in range(num):
        panorama[:,128 + patchSize*(i -1) :128 + patchSize*i] = stack[i]

    # Saving panorama Image
    path = sampleDir + "panorama.png"
    if os.path.exists(path):
        os.remove(path)
    torchvision.utils.save_image(panorama, path) 
    print("[P] Panorama Images has been created as {}".format(path))
if __name__ == '__main__':
    main()