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
