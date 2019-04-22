



from database.seqdb import Seqdb
from libs import utilitylib
from keras.preprocessing import sequence
import numpy as np
import pdb


def getData(SEQUENCE_FILE, configs, filters =[], CRASH_INDEX = "1"):

    seqdb = Seqdb(SEQUENCE_FILE,crashIndex = CRASH_INDEX ); print('Loading data...')
    (xTrain, yTrain), (xTest,yTest) = seqdb.loadData(maxSeqSize = configs['maxSeqSize'],
                                                     testSplit = configs['testSplit'],
                                                     dedup = configs['dedup'],
                                                     filters = filters,
                                                     testFlag = configs.get('testFlag'),
                                                     p2nRatio = configs.get('p2nRatio')) 


    maxlen = seqdb.longestSeqSize   

    xTrain,xTest = seqPadding(xTrain,xTest,maxlen)
    return [xTrain,yTrain,xTest,yTest]


def seqPadding(xTrain,xTest, maxlen):

    '''Pad sequences (samples x time)') '''
    xTrain = sequence.pad_sequences(xTrain, maxlen=maxlen)
    xTest = sequence.pad_sequences(xTest, maxlen=maxlen)
    print('xTrain shape:', xTrain.shape)
    print('xTest shape:', xTest.shape)
    return[xTrain,xTest]


