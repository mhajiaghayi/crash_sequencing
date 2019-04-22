


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
from libs import utilitylib,evallib,modellib,datalib,classifiers,logginglib
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


def precision(y_true, y_pred): 
    """ Precision metric. 
    -    Only computes a batch-wise average of precision.
    """
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) 
    predictedPositive = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positive
    precision = tp / (predictedPositive + K.epsilon()) 
    return precision 


def recall(y_true, y_pred): 
    """ Recall metric. 
    -    Only computes a batch-wise average of recall. 
    -    Computes the recall, a metric for multi-label classification of 
    -    how many relevant items are selected. 
    -    """ 
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # true positive
    possiblePositives = K.sum(K.round(K.clip(y_true, 0, 1))) 
    recall = tp / (possiblePositives + K.epsilon()) 
    return recall 


class History(keras.callbacks.Callback):
    def on_train_begin (self, logs = {}):
        self.data = {'loss':[], 'acc':[], 'prec':[] , 'recall':[],
                    'val_loss':[], 'val_acc':[], 'val_prec':[] , 'val_recall':[] }

    def on_epoch_end (self, batch, logs={}):
        predict = np.asarray(self.model.predict_classes(self.validation_data[0]))
        targ = self.validation_data[1]
        prec= utilitylib.getPrec( predict, targ)
        recall= utilitylib.getRecall( predict, targ)
        accuracy = utilitylib.getAcc(predict,targ)
        print('loss:', logs.get('loss'), ' val_accuracy:', accuracy, ' val_prec:',prec, ' val_recall:', recall, '\n')      
        self.data['loss'].append(logs.get('loss'))
        self.data['acc'].append(logs.get('acc'))
        self.data['prec'].append(logs.get('precision'))
        self.data['recall'].append(logs.get('recall'))
        self.data['val_prec'].append(prec)
        self.data['val_recall'].append(recall)
        self.data['val_acc'].append(accuracy)
        self.data['val_loss'].append(logs.get('val_loss'))
        return


# tensorboard call back to visualize different layers and performance
def getCallbacks(outputFiles, configs):
    history = History()
    embeddingsMetadata = {'embedding': outputFiles['model']['mapping']}
    
    tbCallback = TensorBoard(log_dir= ROOT_DIR,
                        histogram_freq= 100, 
                        write_graph = True,
                        embeddings_freq=100, 
                        batch_size=configs['batchSize'],
                        embeddings_layer_names = ['embedding'],
                        embeddings_metadata= embeddingsMetadata
                         )

    earlyStopping = EarlyStopping(monitor = 'loss',
                              min_delta = 0,
                              patience = 200,
                              verbose = 1,
                              mode = 'min')
    return [history, tbCallback, earlyStopping]


def getModel():
    model = modellib.buildModel(configs)
    adam = optimizers.Adam(lr=.0016,decay= .0001)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=[ 'accuracy', recall, precision]) #  metrics=[ 'accuracy']
    return model



if __name__ == '__main__':
    # Initialize
    seed = 7
    np.random.seed(seed)
    global configs 
    with tf.device('cpu:0'): 
        shutil.rmtree(ROOT_DIR, ignore_errors=True)
          # Flag to show if the data is synthetic or real

        seqdb = Seqdb(SEQUENCE_FILE,crashIndex = CRASH_INDEX ); print('Loading data...')
        (xTrain, yTrain), (xTest, yTest) = seqdb.loadData(maxSeqSize = configs['maxSeqSize'],
                                                         testSplit = configs['testSplit'],
                                                         dedup = configs['dedup'],
                                                         filters = ACTIONS_TO_BE_FILTERED,
                                                         testFlag = configs['testFlag'],
                                                         p2nRatio = configs['p2nRatio'])   
        seqdb.summary();
        configs['maxlen'] = seqdb.longestSeqSize
        configs['actionCount'] = seqdb.actionCount
        configs['trainSize'] = yTrain.size 

        xTrain, xTest = datalib.seqPadding(xTrain, xTest, configs['maxlen'])

        # compare with two classifiers 
        # 1) decision tree without any feature engineering 
        classifiers.decisionTreeClassifier(xTrain,yTrain,xTest,yTest,feMethod="No FE")

        # 2) decision tree with bag of words as classifier 
        (xTrainBW, yTrainBW), (xTestBW,yTestBW) = seqdb.loadBagOfWordsData(testSplit = configs['testSplit'])
        classifiers.decisionTreeClassifier(xTrainBW,yTrainBW,xTestBW,yTestBW,feMethod="Bag of Word")

        logging = logginglib.Logging(seqFileName = SEQUENCE_FILE,
                          perfFileName = r'performance\performance.txt')
        for lstmSize in [6]:
            for embedSize in [3]:
                configs['embeddingSize'] = embedSize
                configs['lstmSize'] = lstmSize
                outputFiles = utilitylib.getOutputFiles(configs)
                seqdb.saveActionMetaData(outputFiles['model']['mapping'])   

                tic = time.time()
                model = getModel()
                model.summary()
                history, tbCallback, earlyStopping = getCallbacks(outputFiles, configs)

                print('Train...')
                model = getModel()
                
                model.fit(xTrain, yTrain,
                          batch_size = configs['batchSize'],
                          epochs = configs['epochs'],
                          validation_data = (xTest, yTest),
                          callbacks = [history, earlyStopping],  # ,tbCallback
                          verbose = 0,
                          class_weight = classWeights)

                toc = time.time() 

                #save the model 
                modellib.saveModel(model,seqdb, outputFiles) 

                # save config files
                modellib.saveConfigParams(configs, outputFiles)

                # calculate roc and plot it 
                print ("roc of testing data ... ")
                evallib.evaluate(xTest,yTest,model,configs['batchSize'])
                yPredProb = model.predict(xTest,batch_size = configs['batchSize'])
                roc = utilitylib.roc(yPredProb, yTest,fileName = outputFiles['roc']); 

                #plot the topology and history (performance vs iterations) of the network
                plot_model(model, to_file = outputFiles['topology'])
                utilitylib.plotHistory(history, outputFiles['hist'])

                logging.finalOutput(roc,configs,tic,toc)
                model.summary(print_fn=logging.model)
