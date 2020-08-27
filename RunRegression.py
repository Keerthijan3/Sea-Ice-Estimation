import sys
import Models.UnetModel as UnetModel
import numpy as np

'''
This is script for training model. Curriculum learning is implemented

INPUTs:

learning - learning rate

step - number of steps before LR decay

num_epochs - total number of epochs per set
(Last set is run for 3x num_epochs)

modelnum - a variable can be set to create a unique
name for the trained model (ex. 1, 2, 3... or a, b, c ..)

traintestsplit - split betweeen train and test data

wdecay - weight decay

directory - data directory. Data must be in .npy file.

OUTPUTs:

Model weights are saved in PreTrained models directory.
'''

learning = sys.argv[1]
step = sys.argv[2]
num_epochs = sys.argv[3]
modelnum = sys.argv[4]
naming = learning.split('-')
traintestsplit = 0.7
wdecay = 0
directory = "DataSets/EnhancementSet_Stages/"
datasetnames = ["stage1", "stage2", "stage3", "stage4"]

print("UNET, CURRICULUM LEARNING (3X LAST SET), 4 STAGES")
print("LEARNING, ", learning)
print("STEPSIZE, ", step)
print("EPOCHS, ", num_epochs)


def GetData(dataset):
    '''
    Input:

    dataset - name of dataset within directory

    Output:
    traindata, testdata, trainlabels, testlabels
    '''
    dataice = np.load(directory + dataset+"ice.npy")
    labelsice = np.load(directory + dataset + "icelabels.npy")
    datawater = np.load(directory + dataset + "water.npy")
    labelswater = np.load(directory + dataset + "waterlabels.npy")
    data = np.concatenate((dataice, datawater), axis=0)
    labels = np.concatenate((labelsice, labelswater), axis=0)
    split = int(traintestsplit*len(labels))
    trainimgs = np.random.choice(len(labels), split, replace=False)
    fullimgs = np.arange(0, len(labels))
    testimgs = np.setdiff1d(fullimgs, trainimgs)
    fullimgs = None
    traindata = data[trainimgs, :, :, :]
    trainlabels = labels[trainimgs, :, :]
    testdata = data[testimgs, :, :, :]
    testlabels = labels[testimgs, :, :]
    return traindata, testdata, trainlabels, testlabels


def GetSingleData(dataset):
    '''

    Loads the data from final stage (marginal ice zone).

    '''
    data = np.load(directory + dataset + '.npy')
    labels = np.load(directory + dataset + 'labels.npy')
    split = int(traintestsplit*len(labels))
    trainimgs = np.random.choice(len(labels), split, replace=False)
    fullimgs = np.arange(0, len(labels))
    testimgs = np.setdiff1d(fullimgs, trainimgs)
    fullimgs = None
    traindata = data[trainimgs, :, :, :]
    trainlabels = labels[trainimgs, :, :]
    testdata = data[testimgs, :, :, :]
    testlabels = labels[testimgs, :, :]
    return traindata, testdata, trainlabels, testlabels


datasetnames = ["stage1", "stage2", "stage3", "stage4"]
s1train, s1test, s1trainlabels, s1testlabels = GetData(datasetnames[0])
s2train, s2test, s2trainlabels, s2testlabels = GetData(datasetnames[1])
s3train, s3test, s3trainlabels, s3testlabels = GetData(datasetnames[2])
s4train, s4test, s4trainlabels, s4testlabels = GetSingleData(datasetnames[3])

testdata = np.concatenate((s1test, s2test, s3test, s4test), axis=0)
testlabels = np.concatenate((s1testlabels, s2testlabels, s3testlabels,
                             s4testlabels), axis=0)
print("TESTDATA GENERATED: ", testdata.shape, ", ", testlabels.shape)

trainset = [s1train, s2train, s3train, s4train]
trainlabelsset = [s1trainlabels, s2trainlabels, s3trainlabels, s4trainlabels]
filesave = None

for i in range(0, 4):
    model = UnetModel.Model()
    if i == 0:
        traindata = trainset[0]
        trainlabels = trainlabelsset[0]
    else:
        model.load(filesave+'.pth')
        traindata = np.concatenate((traindata, trainset[i]), axis=0)
        trainlabels = np.concatenate((trainlabels, trainlabelsset[i]), axis=0)
    print("TRAINING STAGE: ", i)
    print("TRAINING DATA SHAPE: ", traindata.shape, ", ", trainlabels.shape)
    if i == 3:
        model.fit(
                traindata, testdata, trainlabels, testlabels, float(learning),
                decay=0, batch_size=8, epochs=int(num_epochs)*4, momentum=0,
                dampening=0, stepsize=int(step)
                )
    else:
        model.fit(
                traindata, testdata, trainlabels, testlabels, float(learning),
                decay=0, batch_size=8, epochs=int(num_epochs), momentum=0,
                dampening=0, stepsize=int(step)
                )
    filesave = "EnhancementSet_Stages/" + step + "_" + num_epochs + naming[0] + naming[1] + "_" + datasetnames[i] + "_UnetCurrl" + modelnum
    model.save(filesave + ".pth")
    plotsave = (filesave)
    model.plot(plotsave)
f = open("OutputFiles/" + filesave + "trainloss.txt", 'w')
trainloss = [str(loss) for loss in model.trainloss]
testloss = [str(loss) for loss in model.testloss]
trainloss = '\n'.join(trainloss)
f.write(trainloss)
f.close()
f = open("OutputFiles/" + filesave + "testloss.txt", 'w')
testloss = '\n'.join(testloss)
f.write(testloss)
f.close()
