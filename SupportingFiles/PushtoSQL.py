'''
rootdir - location where ground truth files are
preddir - location where predictions are

'''

import EvaluateAllImages
import os

truthdir = "/home/keerthijan/Documents/IceConcentration/IceConcPipeline/DataSets/NewSetNewWater/Evaluation/FINALTESTS/TruthFiles/"
pred_dir = "/home/keerthijan/Documents/IceConcentration/IceConcPipeline/DataSets/NewSetNewWater/Evaluation/FINALTESTS/SQLEvaluation2/"
predfiles = [fil for fil in os.listdir(pred_dir+'/20180227/') if fil.endswith('.npy')]
for predfile in predfiles:
    print(predfile)
    EvaluateAllImages.EvaluateAllInFolderNT2(truthdir, pred_dir, predfile)
    EvaluateAllImages.EvaluateAllInFolder(truthdir, pred_dir, predfile)
    EvaluateAllImages.EvaluateAllInFolderASI(truthdir, pred_dir,predfile)
