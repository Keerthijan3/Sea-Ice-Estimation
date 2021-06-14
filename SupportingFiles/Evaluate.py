'''
This script contains functions required to evaluate predictions
with various ground truth types (ASI, NT2, image analysis charts).
Functions that port the results to an SQL database is provided.
'''


import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mysql.connector


def CalculateMSE(targetloc, predictionloc):
    target = np.load(targetloc)
    target = target[~np.isnan(target)]
    prediction = np.load(predictionloc)
    prediction = prediction[~np.isnan(prediction)]
    MSE = mean_squared_error(target, prediction)
    return MSE


def CalculateL1(targetloc, predictionloc):
    target = np.load(targetloc)
    target = target[~np.isnan(target)]
    prediction = np.load(predictionloc)
    prediction = prediction[~np.isnan(prediction)]
    MAB = mean_absolute_error(target, prediction)
    return MAB


def ReturnMSE(rootdir, predname):
    if rootdir[-1] != '/':
        rootdir = rootdir+'/'
    folders = [folder for folder in os.listdir(rootdir) if folder.startswith('2')]
    keys = []
    values = []
    for folder in folders:
        path = rootdir + folder + '/'
        targetloc = path + 'TargetImage.npy'
        predictionloc = path + predname + '.npy'
        MSE = CalculateMSE(targetloc, predictionloc)
        L1 = CalculateL1(targetloc, predictionloc)
        keys.append(folder+'L1')
        values.append(str(MSE))
        keys.append(folder+'L2')
        values.append(str(L1))
    return keys, values


def EnterToDatabaseIC(keys, values):
    mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="rootpasswordgiven",
            database="verification"
            )
    keys2 = []
    for key in keys:
        k = "`" + key + "`"
        keys2.append(k)
    sqlFormula = "INSERT INTO `verification`.`AggregateEvalIceCharts` (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) VALUES (\"%s\", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    keys = keys2+values
    keys = tuple(keys)
    print("KEYLENGTH: ", len(keys))
    print("KEYS: ", keys)
    mycursor = mydb.cursor()
    mycursor.execute(sqlFormula % keys)
    mydb.commit()


def EnterToDatabaseASI(keys, values):
    mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="rootpasswordgiven",
            database="verification"
            )
    keys2 = []
    for key in keys:
        k = "`" + key + "`"
        keys2.append(k)
    sqlFormula = "INSERT INTO `verification`.`AggregateEvalASI` (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) VALUES (\"%s\", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    keys = keys2+values
    keys = tuple(keys)
    print("KEYLENGTH: ", len(keys))
    print("KEYS: ", keys)
    mycursor = mydb.cursor()
    mycursor.execute(sqlFormula % keys)
    mydb.commit()


def EnterToDatabaseNT2(keys, values):
    mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="rootpasswordgiven",
            database="verification"
            )
    keys2 = []
    for key in keys:
        k = "`" + key + "`"
        keys2.append(k)
    sqlFormula = "INSERT INTO `verification`.`AggregateEvalNT2` (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) VALUES (\"%s\", %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    keys = keys2+values
    keys = tuple(keys)
    print("KEYLENGTH: ", len(keys))
    print("KEYS: ", keys)
    mycursor = mydb.cursor()
    mycursor.execute(sqlFormula % keys)
    mydb.commit()


def EvaluateAllInFolder(truthdir, pred_dir, predfile):
    '''
    icdir - ice chart directory
    pred_dir - prediction directory
    pred_file - prediction file

    '''
    folders = []
    keys = ['Model']
    for i in range(0, 11):
        if i == 0:
            val = '0'
        elif i == 10:
            val = '1'
        else:
            val = str(i/10.0)
        keys.append(val+'_L1')
        keys.append(val+'_L2')
        keys.append(val+'_StdDev')
        keys.append(val+'_Bias')
        keys.append(val+'_NumPoints')

    vals = [predfile]
    for i in range(0, 11):
        val = i/10
        for idx, folder in enumerate(folders):
            truthfile = np.load(truthdir+'/'+folder+'/icechartvis.npy')
            pred = np.load(pred_dir+'/'+folder+'/'+'/' + predfile)
            truthfile[np.isnan(pred)] = np.nan
            pred[np.isnan(truthfile)] = np.nan
            if idx == 0:
                allpoints = pred[truthfile == val]
            else:
                allpoints = np.concatenate((allpoints, pred[truthfile == val]))
        if len(allpoints) > 10:
            trutharray = val*np.ones(np.shape(allpoints))
            bias = np.mean(allpoints)
            L1 = mean_absolute_error(trutharray, allpoints)
            L2 = mean_squared_error(trutharray, allpoints)
            stddev = np.std(allpoints)
        else:
            trutharray = []
            bias = -1
            L1 = -1
            L2 = -1
            stddev = -1
        vals.append(str(L1))
        vals.append(str(L2))
        vals.append(str(stddev))
        vals.append(str(bias))
        vals.append(str(len(trutharray)))
    EnterToDatabaseIC(keys, vals)


def EvaluateAllInFolderASI(truthdir, pred_dir, predfile):
    '''
    icdir - ice chart directory
    pred_dir - prediction directory
    pred_file - prediction file
    Change ASI to NT2 when necessary

    '''
    folders = []
    keys = ['Model']
    for i in range(0, 10):
        if i == 0:
            key = '0-0.1'
        elif i == 9:
            key = '0.9-1'
        else:
            val = i/10.0
            key = "{:.1f}".format(val)+'-'+"{:.1f}".format(val+0.1)
        keys.append(key+'_L1')
        keys.append(key+'_L2')
        keys.append(key+'_StdDev')
        keys.append(key+'_Bias')
        keys.append(key+'_NumPoints')

    vals = [predfile]
    for i in range(0, 10):
        val = i/10
        for idx, folder in enumerate(folders):
            truthfile = np.load(truthdir+'/'+folder+'/ASI.npy')
            pred = np.load(pred_dir+'/'+folder+'/'+'/' + predfile)
            truthfile[np.isnan(pred)] = np.nan
            pred[np.isnan(truthfile)] = np.nan
            if i == 0:
                if idx == 0:
                    allpoints = pred[np.where(np.logical_and(truthfile >= val, truthfile <= val+0.1))]
                    trutharray = truthfile[np.where(np.logical_and(truthfile >= val, truthfile <= val + 0.1))]
                else:
                    allpoints = np.concatenate((allpoints, pred[np.where(np.logical_and(truthfile >= val, truthfile <= val+0.1))]))
                    trutharray = np.concatenate((trutharray, truthfile[np.where(np.logical_and(truthfile >= val, truthfile <= val+0.1))]))
            else:
                if idx == 0:
                    allpoints = pred[np.where(np.logical_and(truthfile > val, truthfile <= val+0.1))]
                    trutharray = truthfile[np.where(np.logical_and(truthfile > val, truthfile <= val+0.1))]
                else:
                    allpoints = np.concatenate((allpoints, pred[np.where(np.logical_and(truthfile > val, truthfile <= val+0.1))]))
                    trutharray = np.concatenate((trutharray, truthfile[np.where(np.logical_and(truthfile > val, truthfile <= val+0.1))]))
        if len(allpoints) > 10:
            bias = np.mean(allpoints)
            L1 = mean_absolute_error(trutharray, allpoints)
            L2 = mean_squared_error(trutharray, allpoints)
            stddev = np.std(allpoints)
        else:
            trutharray = []
            bias = -1
            L1 = -1
            L2 = -1
            stddev = -1
        vals.append(str(L1))
        vals.append(str(L2))
        vals.append(str(stddev))
        vals.append(str(bias))
        vals.append(str(len(trutharray)))
    EnterToDatabaseASI(keys, vals)


def EvaluateAllInFolderNT2(truthdir, pred_dir, predfile):
    '''
    icdir - ice chart directory
    pred_dir - prediction directory
    pred_file - prediction file

    '''
    folders = []
    keys = ['Model']
    for i in range(0, 10):
        if i == 0:
            key = '0-0.1'
        elif i == 9:
            key = '0.9-1'
        else:
            val = i/10.0
            key = "{:.1f}".format(val)+'-'+"{:.1f}".format(val+0.1)
        keys.append(key+'_L1')
        keys.append(key+'_L2')
        keys.append(key+'_StdDev')
        keys.append(key+'_Bias')
        keys.append(key+'_NumPoints')

    vals = [predfile]
    for i in range(0, 10):
        val = i/10
        for idx, folder in enumerate(folders):
            truthfile = np.load(truthdir+'/'+folder+'/NT2.npy')
            pred = np.load(pred_dir+'/'+folder+'/'+'/' + predfile)
            truthfile[np.isnan(pred)] = np.nan
            pred[np.isnan(truthfile)] = np.nan
            if i == 0:
                if idx == 0:
                    allpoints = pred[np.where(np.logical_and(truthfile >= val, truthfile <= val + 0.1))]
                    trutharray = truthfile[np.where(np.logical_and(truthfile >= val, truthfile <= val + 0.1))]
                else:
                    allpoints = np.concatenate((allpoints, pred[np.where(np.logical_and(truthfile >= val, truthfile <= val + 0.1))]))
                    trutharray = np.concatenate((trutharray, truthfile[np.where(np.logical_and(truthfile >= val, truthfile <= val + 0.1))]))
            else:
                if idx == 0:
                    allpoints = pred[np.where(np.logical_and(truthfile > val, truthfile <= val + 0.1))]
                    trutharray = truthfile[np.where(np.logical_and(truthfile > val, truthfile <= val + 0.1))]
                else:
                    allpoints = np.concatenate((allpoints, pred[np.where(np.logical_and(truthfile > val, truthfile <= val + 0.1))]))
                    trutharray = np.concatenate((trutharray, truthfile[np.where(np.logical_and(truthfile > val, truthfile <= val + 0.1))]))
        if len(allpoints) > 10:
            bias = np.mean(allpoints)
            L1 = mean_absolute_error(trutharray, allpoints)
            L2 = mean_squared_error(trutharray, allpoints)
            stddev = np.std(allpoints)
        else:
            trutharray = []
            bias = -1
            L1 = -1
            L2 = -1
            stddev = -1
        vals.append(str(L1))
        vals.append(str(L2))
        vals.append(str(stddev))
        vals.append(str(bias))
        vals.append(str(len(trutharray)))
    EnterToDatabaseNT2(keys, vals)
