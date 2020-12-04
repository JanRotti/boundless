import numpy as np
import time
import sys
import torch
import torchvision
from subprocess import call
from torch.autograd import grad, Variable
import matplotlib.pyplot as plt

def plotLosses(lossesList, legendsList, fileOut):
    assert len(lossesList) == len(legendsList)
    for i, loss in enumerate(lossesList):
        plt.plot(loss, label=legendsList[i])
    plt.legend()
    plt.savefig(fileOut)
    plt.close()        

def saveImage(img, nrow, epoch, step, sampleDir, saveTime = time.ctime(),name = None):
    filename = 'model_{}_epoch_{}_step_{}_time_{}.png'.format(name, epoch, step, saveTime)
    filePath = sampleDir + filename
    torchvision.utils.save_image(img, filePath, nrow)
    
def denormalize(img):
    # denormalize image tensor from [-1,1] to [0,1]
    return (img+1) / 2
  
def getDevice(gpuNum = 0):
    gpuNum = 0 if torch.cuda.device_count() < gpuNum else gpuNum   
    device = ('cuda:' + str(gpuNum)) if torch.cuda.is_available() else 'cpu'
    return device