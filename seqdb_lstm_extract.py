


'''
coder: Mahdi Hajiaghayi, Ehsan Vahedi
date: April 2017
summary: Given a trained LSTM model and dataset, this code attempts to extract the 
contributors as well as blockers. 

'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding,LSTM
from keras.callbacks import ModelCheckpoint
import keras
import tensorflow as tf
from keras.models import load_model
import keras.backend as K
from keras.utils import np_utils
from database.seqdb import Seqdb
from database.seqdb import removeImmediateDuplicate
import numpy as np 
import pdb,os
from libs import utilitylib, modellib,datalib,evallib
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.callbacks import TensorBoard
import shutil
from libs.extract_reportlib import ReportSummary
from _init_global_vars_extract import *

class Sequence:
    def __init__(self,_input, 
                label = None, 
                isEventEncoded = False,
                eraseWithZero = False,
                isDedup = True,
                configs = None ):
        ''' 
        each seq has two key members: 
        self.arr is a raw input date in a list form
        self.vector is the vectorized version of self.arr plus seq padding. so the lenght of self.arr and self.vector
        may differ. 
        in case of eventisEncoded= True,
        self.arr = self.vector. note when we read data from the seqDb, everything is encoded already.
        ''' 

        self.configs = configs
        self.label = label 
        self.eraseWithZero = eraseWithZero   #instead of shortening the sequence, we replace the event with zero
        self.isEventEncoded = isEventEncoded
        try: 
            if self.isEventEncoded:
                self.vector = np.array([_input])
                self.arr = _input

            else:
                if isinstance(_input,str): 
                    self.arr = _input.lower().split(',')
                else: 
                    self.arr = _input
            if self.configs['dedup']:
                self.arr = removeImmediateDuplicate(self.arr)
            self.vector = self.vectorize(self.arr)
        except:
            print("if your inputs are raw data, please set isEventEncoded=False")
            raise

        self.contributors = list()
        self.contributorIds = list()
        self.blockers = list()
        self.blockerIds = list()

    def setId(self,_id):
        self.id = _id 


    def vectorize(self, seq,oov_char=2):
        # read the events in the given sequence and converts them to integer via the config's vocabulary
        # if it is not encoded already. Then it adds the sequence padding. 
        # the output is ready to be fed to the model.   
        # 
        vec = list()
        # pdb.set_trace()
        for event in seq:
            if self.isEventEncoded:
                if self.eraseWithZero and event ==0:
                    vec.append(0)
                else:
                    vec.append(event)
            else:
                if self.eraseWithZero and event == 0:
                    vec.append(0)
                else:
                    vec.append(self.configs['vocab'].get(event,oov_char))
        return sequence.pad_sequences([vec], maxlen=self.configs['maxSeqSize'])


    def eraseEventK(self,k, importantIds):
        eventK = self.arr[k]
        seqArrMuted = list()
        for j, event in enumerate(self.arr):
            if (event != eventK  or j > k or j in importantIds):
                seqArrMuted.append(event)
            # insert 0 if eraseWithZero is enabled.
            elif (event == eventK  and j <= k and self.eraseWithZero): 
               seqArrMuted.append(0)
            
        return seqArrMuted



    def extractImportantEvents(self, model, diffThreshold = .25, confidence = 0): 
        # This function extracts crash contributor and blockers based on the seq.vector. 
        # contributors: removing them, change the predictor from 1 to 0 
        # blocker: removing them, change the predictor from 0 to 1 
        self.importantIds = list()
        self.eventsEffect = list()

        self.pred = model.predict_classes(self.vector, verbose = False)[0][0] 
        self.prob = float("%.3f"%model.predict(self.vector)[0][0])  
        self.conf = self.prob * self.pred + (1-self.prob)*(1-self.pred)
        if self.conf > confidence:
            for k , event in enumerate(self.arr):
                seqArrMuted = self.eraseEventK(k,self.importantIds)
                vectorMuted = self.vectorize(seqArrMuted)

                probMuted = model.predict(vectorMuted)
                predMuted = model.predict_classes(vectorMuted,verbose = False)[0][0]
                predDiff =  predMuted - self.pred
                probDiff = float("%.3f"%(probMuted - self.prob ))
                if (predDiff != 0 or abs(probDiff) > diffThreshold ) : self.importantIds.append(k)
                if (predDiff > 0 or probDiff > diffThreshold): self.blockerIds.append(k); self.blockers.append(event)
                if (predDiff < 0 or probDiff < - diffThreshold): self.contributorIds.append(k); self.contributors.append(event)
                self.eventsEffect.append(abs(probDiff)) 
        return [self.contributors,self.blockers]



if __name__ == '__main__':

    with tf.device('cpu:0'):
        EVENT_ENCODED =  True
        configs = modellib.getConfigs(CONFIG_FILE)
        model = modellib.loadModel(configs)
        trash1,trash2,inputs,labels = datalib.getData(SEQUENCE_FILE,configs,ACTIONS_TO_BE_FILTERED,CRASH_INDEX)
        evallib.evaluate(inputs[0:200], labels[0:200], model, configs['batchSize'])
        # inputs = [r'a,f,b,c,e,f', r'c,a,f,h,f,c,e,c,k,b,f,a,b,j,e', r'a,f,b,c,a', r'g,b,g,a,c,c,a,f,b,c,k,b,c,f,c',r'g,b,b,d,f,g,f,f,f,i,i,g,b,c,c',
        #         r'f,h,a,a,d,b,d,h,f,c,g,b,j,d,d',r'k,f,b,c,j,b,h,f,f,c,f,c,b,f,c',r'h,b,j,c,a,k,c,d,c,f,b,c,i,d',r'f,c,d,b,l,g,l,c,i,i,c,b,f,a,b'] 
        # labels = [0,0,1,1,1,1,1,1,1]
        seqs = list()
        report = ReportSummary(vocab = configs['vocab'],isEventEncoded= EVENT_ENCODED)
        for _id, _input in enumerate(inputs[0:5000]): 
            seq = Sequence(_input, labels[_id], 
                            configs = configs,
                            isEventEncoded = EVENT_ENCODED,
                            eraseWithZero = False)
            print("id:",_id)
            seq.setId(_id)
            seqs.append(seq)
            contributors, blockers = seq.extractImportantEvents(model, diffThreshold=.40, confidence = .8)
            report.add(contributors= contributors, blockers = blockers)
            
        report.saveHTML(seqs, htmlFile= HTML_OUTPUT_FILE)
        report.sort(60)
        report.save(REPORT_OUTPUT_FILE)