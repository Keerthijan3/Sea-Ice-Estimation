'''
This script is used to make predictions.

Input:

ModelWeights - Reads model weights from PreTrainedModels folder

rootdir - Directory consisting of folders with scenes. Script
searches for HH/HV images and ASI NETCDF file. ASI file is used
for landmask.

Output:
Predictions are made and stored in the same location as the
scenes.

'''


import SupportingFiles.LoadData as LoadData
import numpy as np
import os
import Models.UnetModel as UnetModel
import sys
import matplotlib.pyplot as plt
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ModelWeights = sys.argv[1]
rootdir = ""
model = UnetModel.Model()
model.load(ModelWeights)
naming = ModelWeights.split('.')
folders = [folder for folder in os.listdir(rootdir) if folder.startswith('2')]
for folder in folders:
    PredictImage = rootdir + folder + '/'
    HH, HV = LoadData.LoadImage(PredictImage)
    HH = HH.astype(np.float32)/255 - 0.5
    HV = HV.astype(np.float32)/255 - 0.5
    target, nans = LoadData.LoadTruth(PredictImage)
    height, width = np.shape(HH)
    img = np.stack((HH, HV), axis=0)
    predmatrix = np.zeros((height, width))
    img = torch.from_numpy(img)
    img = img.unsqueeze(dim=0).to(device)
    out = model.predict(img)
    out = out[0, 0, :, :]
    out = out.cpu().numpy()
    out[np.isnan(target)] = np.nan
    out[out > 1] = 1
    out[out < 0] = 0
    filesave = PredictImage+naming[0]+'_OUT'
    np.save(filesave+'.npy', out)
    plt.imsave(filesave+'.tif', out, vmin=0, vmax=1)
