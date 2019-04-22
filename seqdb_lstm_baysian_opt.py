

'''
coder: Mahdi Hajiaghayi, Ehsan Vahedi
date: April 2017
summary: Trains a LSTM on the activity or tcid sequences that are saved in db called Seqdb.
here, we want to know whether we can predict the crash based on the sequence data. 
if we can, then we are able to extract the pattern from the sequence data. 

'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf
from keras.models import load_model
# from keras import metrics
import keras.backend as K
from keras.utils import np_utils
from database.seqdb import Seqdb
# from database.psudo_seqdb import PsudoSeqdb
import numpy as np 
import pdb,os
from libs import utilitylib,evallib,modellib,datalib,classifiers
from keras.utils import plot_model
from keras.callbacks import TensorBoard,EarlyStopping
import shutil,time
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.metrics import recall_score
from _init_global_vars_train import *
from bayes_opt import BayesianOptimization


def getModel():
    model = modellib.buildModel(configs)
    adam = optimizers.Adam(lr=configs['learningRate'],decay= .0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[ 'accuracy']) #  metrics=[ 'accuracy']
    return model

def evaluateLSTM(embeddingSize,
                lstmSize,
                lr,
                networkType):
    configs['embeddingSize'] = int(embeddingSize)
    configs['lstmSize'] = int(lstmSize)
    configs['learningRate'] = lr 
    if int(networkType) == 1:
      configs['networkType'] = "bidrectional"
    else:
      configs['networkType'] = "lstm"

    modelCV = KerasClassifier(build_fn=getModel,
                          epochs=configs['epochs'], 
                          batch_size=configs['batchSize'],
                          verbose=0)
    kfold = KFold(n_splits=5, random_state=seed)
    scoring = ['precision', 'recall','accuracy','roc_auc','f1']
    results = cross_validate(modelCV, xTrain, yTrain, cv=kfold, scoring = scoring, return_train_score=False)
    return np.mean(results['test_f1'])


if __name__ == '__main__':
    # Initialize
    seed = 7
    np.random.seed(seed)
    global configs 
    with tf.device('cpu:0'): 
        shutil.rmtree(ROOT_DIR, ignore_errors=True)
          # Flag to show if the data is synthetic or real

        seqdb = Seqdb(SEQUENCE_FILE,crashIndex = CRASH_INDEX ); print('Loading data...')
        (xTrain, yTrain), (xTest,yTest) = seqdb.loadData(maxSeqSize = configs['maxSeqSize'],
                                                         testSplit = configs['testSplit'],
                                                         dedup = configs['dedup'],
                                                         filters = ACTIONS_TO_BE_FILTERED,
                                                         testFlag = configs['testFlag'],
                                                         p2nRatio = configs['p2nRatio'])   


        seqdb.summary()
        configs['maxlen'] = seqdb.longestSeqSize
        configs['actionCount'] = seqdb.actionCount
        configs['trainSize'] = yTrain.size 

        xTrain,xTest = datalib.seqPadding(xTrain,xTest,configs['maxlen'])

        lstmBO = BayesianOptimization(evaluateLSTM, {'embeddingSize' : (1,10),
                                                     'lstmSize' : (4,20),
                                                     'lr' : (.001,.03),
                                                     'networkType' :(.5,1.5)})
        lstmBO.explore({ 'embeddingSize' : [3 ,8 ,2],
                         'lstmSize' : [6,14,6],
                         'lr' : [.01,.02,.001],
                         'networkType' : [.5,.5,1]})
        lstmBO.maximize(n_iter=20,acq= 'ei')
        print(lstmBO.res['max'])


