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
import shutil
import os
from utils.parser import *

def main():
	args = parameter_parser()
	name = args.runName
	sampleDir = './samples/'+ name +'/'
	checkpointDir = './checkpoints/'+ name +'/'
	pycacheDir = './__pycache__/'
	if os.path.exists(sampleDir):
		shutil.rmtree(sampleDir)
	if os.path.exists(checkpointDir):
		shutil.rmtree(checkpointDir)
	if os.path.exists(pycacheDir):
		shutil.rmtree(pycacheDir)
if __name__ == '__main__':
	main()
