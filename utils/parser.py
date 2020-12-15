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
import argparse

def parameter_parser():

    parser = argparse.ArgumentParser(description = "GPR WGAN")

    parser.add_argument("--epochs",dest = "epochs",type = int, default = 10000, help = "Number of gradient descent iterations. Default is 200.")

    parser.add_argument("--dataDir",dest = "dataDir",type = str, default = "./data", help = "Directory with files to be processed")
    
    parser.add_argument("--maskRatio",dest = "maskRatio",type = float, default = 0.25, help = "Ratio of images to be reconstructed")

    parser.add_argument("--learningRate",dest = "learningRate",type = float,default = 0.0001, help = "Gradient descent learning rate. Default is 0.0001.")			 

    parser.add_argument("--batchSize",dest = "batchSize",type = int, default = 64,help = "Batch size")

    parser.add_argument("--advLambda",dest = "advLambda", type = float,default = 10, help = "Weight of adversarial Loss")

    parser.add_argument("--imgSize",dest = "imgSize",type = int,default = 128, help = "Img Size to Crop to")
    
    parser.add_argument("--runName",dest = "runName",type = str ,default= "test", help="Name for output files")

    parser.add_argument("--sampleFile",dest = "sampleFile",type = str ,default= "Lena.jpg", help="File to perform panorama extension on")

    parser.add_argument("--sampleEpoch",dest = "sampleEpoch",type = int ,default= 10000, help="Model Epoch to Load")

    parser.add_argument("--num",dest = "num",type = int ,default = 1, help = "Number of Extension Steps to be performed")
    
    return parser.parse_args()
