#############################################################################
#                                                                           #
#   Synthetic GPR Image Generation using Generative Adversarial Networks    #
#   Copyright (C) 2020  Jan Rottmayer                                       #
#                                                                           #
#   This program is free software: you can redistribute it and/or modify    #
#   it under the terms of the GNU General Public License as published by    #
#   the Free Software Foundation, either version 3 of the License, or       #
#   (at your option) any later version.                                     #
#                                                                           #
#   This program is distributed in the hope that it will be useful,         #  
#   but WITHOUT ANY WARRANTY; without even the implied warranty of          #
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           #
#   GNU General Public License for more details.                            #
#                                                                           #
#   You should have received a copy of the GNU General Public License       #
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.  #
#                                                                           #
#############################################################################
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