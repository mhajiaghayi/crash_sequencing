
'''
coder: Mahdi Hajiaghayi, Ehsan Vahedi
date: April 2017
summary: Load a trained model and predict the output for given sequences.

'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow
from keras.models import load_model
# from keras import metrics
import keras.backend as K
from keras.utils import np_utils
from database.seqdb import Seqdb
from database.psudo_seqdb import PsudoSeqdb
import numpy as np 
import pdb,os
from libs import utilitylib,modellib,datalib,evallib
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import shutil






def predict(model, seqs, vocab):
    if isinstance(seqs,str):
        seqs = [seqs]
    for seq in seqs:
        xTest = vectorize(seq,vocab)
        yTrainPred = model.predict_classes(xTest, verbose = False)
        yTrainProb = model.predict(xTest)
        print('\n for seq {0} model.predict: {1} with prob {2}'.format(*[seq,yTrainPred,yTrainProb]))
    return

def vectorize(text, vocab, 
              maxlen=14, start_char=1, oov_char=2, index_from=3):
    """ might not be consistent with vectorize_data. """
    if isinstance(text, str):
        text = [text]
    v = [[vocab.get(w, oov_char) for w in t.lower().split(',')] for t in text]
    return sequence.pad_sequences(v, maxlen=maxlen)


if __name__ == '__main__':

    SEQUENCE_FILE  = "data\\test_action_Ehsan.txt"
    CONFIG_FILE = "models\configs_actions12_trains5000_maxlen14_embedding4_epochs_1000_conv1d_0.tsv"
    pdb.set_trace()
    configs = modellib.getConfigs(CONFIG_FILE)
    model = modellib.loadModel(configs)
    xTrain,yTrain,inputs,labels = datalib.getData(SEQUENCE_FILE,configs)



    seqs = [r'a,f,b,c,e,f', r'a,f,b,c,a', r'g,b,g,a,c,c,a,f,b,c,k,b,c,f,c',r'g,b,b,d,f,g,f,f,f,i,i,g,b,c,c',
            r'f,h,a,a,d,b,d,h,f,c,g,b,j,d,d',r'k,f,b,c,j,b,h,f,f,c,f,c,b,f,c',r'h,b,j,c,a,k,c,d,c,f,b,c,i,d'] 
    predict(model = model, 
                seqs = seqs, 
                vocab=configs['vocab'])
    ypred = model.predict_classes(xTrain)
    evallib.evaluate(xTrain,yTrain,model,configs['batchSize'])

   

