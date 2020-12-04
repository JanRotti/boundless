import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import argparse

# Preprocessing
import glob
from PIL import Image, ImageOps
import random

# model dependencies
import time
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad, Variable
from utils.parser import *
from utils.utils import *
from utils.dataset import *
from models import Generator, Discriminator, InceptionExtractor

	

class Boundless():
    def __init__(self, batchSize=64, learningRate=0.0001, epochs=10,\
        dataDir= './data', advLambda = 10, runName = "test", imgSize = 128, maskRatio = 0.25):
        self.dataDir = dataDir
        self.imgSize = imgSize
        self.maskRatio = maskRatio
        self.useCuda = True if torch.cuda.is_available() else False
        self.batchSize = batchSize
        self.learningRate = learningRate
        self.epochs = epochs
        self.advLambda = advLambda
        self.checkpointDir = './checkpoints/' + runName + "/"
        self.sampleDir = './samples/' + runName + "/"
        self.dataLoader = ""
        if not os.path.exists(self.checkpointDir):
            os.makedirs(self.checkpointDir)
        if not os.path.exists(self.sampleDir): 
            os.makedirs(self.sampleDir)
        self.filePathG = self.checkpointDir + "G_state_{}.pth".format(0)
        self.filePathD = self.checkpointDir + "D_state_{}.pth".format(0)
        self.device = getDevice(2)
        self.buildModel()
        self.loadData()
        self.pixelLoss = nn.L1Loss().to(self.device)
        self.contentLoss = nn.L1Loss().to(self.device)
        
    def loadData(self):
        self.dataLoader = makeDataLoader(self.dataDir, self.batchSize, self.imgSize, self.maskRatio)
        print("[*] Data has been loaded successfully.")
        
    def buildModel(self, optimizer = "adam"):
        self.G = Generator()
        self.D = Discriminator()
        self.I = InceptionExtractor()
        self.I.eval() # dont train IneceptionExtractor
        
        if self.useCuda:
            self.G.to(self.device)
            self.D.to(self.device)
            self.I.to(self.device)
        if optimizer == "adam":
            self.gOptim = optim.Adam(self.G.parameters(),lr=self.learningRate, betas=(0.0, 0.9))
            self.dOptim = optim.Adam(self.D.parameters(),lr=4*self.learningRate, betas=(0.0, 0.9))
        elif optimizer == "rms":
            self.gOptim = optim.RMSprop(self.G.parameters(),lr=self.learningRate)
            self.dOptim = optim.RMSprop(self.D.parameters(),lr=4*self.learningRate)
        else:
            print("Unrecognized Optimizer !")
            return 0
        
        
    def saveModel(self, epoch):
        self.filePathG = self.checkpointDir + "G_state_{}.pth".format(epoch)
        self.filePathD = self.checkpointDir + "D_state_{}.pth".format(epoch)
        torch.save(self.D.state_dict(), self.filePathD)
        torch.save(self.G.state_dict(), self.filePathG)
    
    def loadModel(self, directory = ''):
        if not directory:
            directory = self.checkpointDir
        listG = glob.glob(directory + "G*.pth")
        listD = glob.glob(directory + "D*.pth")
        if  len(listG) == 0 or len(listD) == 0:
            print("[*] No Checkpoint found! Starting from scratch.")
            return 1
        Gfile = max(listG, key=os.path.getctime)
        Dfile = max(listD, key=os.path.getctime)
        epochFound = int( (Gfile.split('_')[-1]).split('.')[0])
  
        print("[*] Checkpoint {} found at {}!".format(epochFound, directory))
        dState = torch.load(Dfile)
        gState = torch.load(Gfile)
        
        self.D.load_state_dict(dState)
        self.G.load_state_dict(gState)
        return epochFound
    
    def logProcess(self, epoch, step, stepPerEpoch, gLog, dLog, advLoss, gPixLoss, lossesF):
        summaryStr = 'Epoch [{}], Step [{}/{}], Losses: G [{:4f}], D [{:4f}], adv [{:4f}], rec: [{:4f}]'.format(epoch, step, stepPerEpoch, gLog, dLog, advLoss, gPixLoss)
        print(summaryStr)
        lossesF.write(summaryStr)
    
    def plot_losses(lossesList, legendsList, fileOut):
        assert len(lossesList) == len(legendsList)
        for i, loss in enumerate(lossesList):
            plt.plot(loss, label=legendsList[i])
        plt.legend()
        plt.savefig(fileOut)
        plt.close()
    
    def sample(self, epoch, gStacked, fakeImg, realImg, maskedImg):
        imgGrid = (torch.cat((realImg, maskedImg, gStacked, fakeImg),-1) +1) / 2 # Denormalize Img Grid
        path = self.sampleDir + "sampled_{}.png".format(epoch)
        torchvision.utils.save_image(imgGrid, path, nrow=1)
        time.sleep(2)
        self.G.train()
    
    def discriminatorStep(self, realImg, mask , classLabels, gStacked):
        self.dOptim.zero_grad()
        
        # Score Images
        realScore = self.D(realImg, mask, classLabels)
        fakeScore = self.D(gStacked.detach(), mask, classLabels)

        # Discriminator Losses 
        realLoss = nn.ReLU()(1 - realScore).mean()
        fakeLoss = nn.ReLU()(1 + fakeScore).mean()
        dLoss = realLoss + fakeLoss

        # Optimization Step        
        dLoss.backward()              # Calculate Gradients trough Backpropagation
        self.dOptim.step()            # Optimization Step
        
        return dLoss.item(), realLoss.item(), fakeLoss.item()
    
    def generatorStep(self, stacked, realImg, mask, maskedImg, classLabels):
        self.gOptim.zero_grad()
        
        # Reconstruct Images
        fakeImg = self.G(stacked)
        
        # score Reconstruction
        gPixLoss = self.pixelLoss(fakeImg, realImg)
        
        # Combine RealImage Section with Generated Mask Region
        gStacked = fakeImg * mask + maskedImg
        
        # Score Fake Images
        fakeScore = self.D(gStacked, mask, classLabels)
        
        advLoss = - fakeScore.mean()
        gLoss = self.advLambda * advLoss + gPixLoss
        gLoss.backward()
        self.gOptim.step()
        
        return fakeImg, gStacked, gLoss.item(), advLoss.item(), gPixLoss.item()
        
    def trainModel(self, loadDir = ""):
        self.epoch = self.loadModel(loadDir)
        lossesF = open(self.checkpointDir + "losses.txt",'a+')
        # Updated Lossing Lists -> Adapt rest of Code!!!
        gLosses, advLosses, gPixLosses = [],[],[] 
        dLosses, fakeLosses, realLosses = [],[],[]
        stepPerEpoch = len(self.dataLoader)
        
        # Initialize Losses due to discUpdates

        gLoss, advLoss, gPixLoss = 0., 0., 0.
        for epoch in range(self.epoch,self.epochs):
            for batch, images in enumerate(self.dataLoader):
                maskedImg = Variable(images["lr"].to(self.device))
                realImg = Variable(images["hr"].to(self.device))
                mask = Variable(images["alpha"].to(self.device))
                stacked = Variable(images["clip"].to(self.device))
                classLabels = self.I(realImg).detach()
                
                #                     #
                #   Train  Generator  #
                #                     #    
                fakeImg, gStacked, gLoss, advLoss, gPixLoss = self.generatorStep(stacked, realImg, mask, maskedImg, classLabels)
                
                #                     #
                # Train Discriminator #
                #                     #
                
                dLoss, realLoss, fakeLoss = self.discriminatorStep(realImg, mask , classLabels, gStacked)
                
                # log Losses to Lists
                dLosses.append(dLoss)
                gLosses.append(gLoss)
                fakeLosses.append(fakeLoss)
                realLosses.append(realLoss)
                advLosses.append(advLoss)
                gPixLosses.append(gPixLoss)
                
                if batch % 8 == 0:
                    self.logProcess(epoch, batch, stepPerEpoch, gLosses[-1], dLosses[-1], advLosses[-1], gPixLosses[-1], lossesF)

            if epoch % 10 == 0:
                plotLosses([gLosses, dLosses], ["gen", "disc"], self.sampleDir + "losses.png")
                plotLosses([advLosses,gPixLosses], ["adversarial","reconstruction"], self.sampleDir + "g_loss_components.png")
                plotLosses([fakeLosses, realLosses],["d_fake", "d_real"], self.sampleDir + "d_loss_components.png")
            
            if epoch == 1:
                saveImg = denormalize(realImg)
                torchvision.utils.save_image(saveImg,os.path.join(self.sampleDir,'real.png'),nrow = 10)     

            if epoch % 100 == 0:
                self.saveModel(epoch)
                self.sample(epoch, gStacked, fakeImg, realImg, maskedImg)
            
def main():
    args = parameter_parser()
    model = Boundless(batchSize=args.batchSize, learningRate=args.learningRate, epochs=args.epochs,\
                    dataDir= args.dataDir,advLambda = args.advLambda,
                    runName = args.runName, imgSize = args.imgSize, maskRatio = args.maskRatio)
    
    model.trainModel()

if __name__ == '__main__':
    main()
