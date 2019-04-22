

import numpy as np 
from keras.models import model_from_json
import pdb,os
from libs import utilitylib
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D
import keras.backend as K


def getModelFiles(configs):
    if configs.get('conv1D') != None:
        configStr = 'actions{actionCount}_trains{trainSize}_maxlen{maxlen}_embedding{embeddingSize}_epochs_{epochs}_conv1d_{conv1D}_lstmsize_{lstmSize}'.format(**configs)
    else:
        configStr = 'actions{actionCount}_trains{trainSize}_maxlen{maxlen}_embedding{embeddingSize}_epochs_{epochs}_lstmsize_{lstmSize}'.format(**configs)

    modelFiles = {'weights': os.path.join('models','weight_' + configStr + '.h5'),
              'json': os.path.join('models','json_model_' + configStr + '.txt'),
              'mapping': os.path.join('models','mapping_' + configStr + '.tsv')
                                }
    return modelFiles                 

def getVocabFile (configs):
    if configs.get('conv1D') != None:
        configStr = 'actions{actionCount}_trains{trainSize}_maxlen{maxlen}_embedding{embeddingSize}_epochs_{epochs}_conv1d_{conv1D}_lstmsize_{lstmSize}'.format(**configs)
    else:
        configStr = 'actions{actionCount}_trains{trainSize}_maxlen{maxlen}_embedding{embeddingSize}_epochs_{epochs}_lstmsize_{lstmSize}'.format(**configs)
    return os.path.join('models','mapping_' + configStr + '.tsv')


def loadModel(configs):
    files = getModelFiles(configs)
    with open(files['json'],'r') as f:
        jsonModel = f.readlines()[0]

    model = model_from_json(jsonModel)

    # load weights 
    model.load_weights(files['weights'])
    return model 


def getConfigs(CONFIG_FILE):
    configs = {}
    with open(CONFIG_FILE,'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        k , v = line.replace('\n','').split('\t')
        if (k in ['testSplit','p2nRatio']):
            configs[k] = float(v)
        elif (k == 'testFlag'):
            configs[k] = int( v == 'True')
        elif (k.lower() == 'networktype'):
            configs[k] = int( v == 'LSTM')        
        else: 
            configs[k]= int(v)
    vocabFile = getVocabFile(configs)
    configs['vocab'] = utilitylib.readVocab(vocabFile)
    return configs


def saveModel(model, seqdb, outputFiles):
    # save the weights and architecture separately
    # save architecture
    jsonModel = model.to_json()
    with open(outputFiles['model']['json'],'w') as f:
        f.write(jsonModel)

    # save weights 
    model.save_weights(outputFiles['model']['weights'])

    # save the mapping
    seqdb.saveActionMetaData(outputFiles['model']['mapping'])
    return 

def saveConfigParams(configs,outputFiles):
    with open(outputFiles['model']['configs'],'w') as file:
        count = 0 
        for k,v in configs.items():
            count = count + 1
            if count < len(configs.keys()):
                file.write(k +'\t' + str(v) + '\n')
            else:
                file.write(k +'\t' + str(v))



def buildModel(configs):
    print('Build model...')
    K.set_learning_phase(0)
    model = Sequential()
    model.add(Embedding(configs['actionCount']+1, configs['embeddingSize'], name = 'embedding'))
    if (configs.get('networkType') == "Conv1D"):
        model.add(Conv1D(configs['lstmSize'], kernel_size = 3, padding= 'same', activation = 'relu'))
        model.add(MaxPooling1D(pool_size =2))
        model.add(LSTM(configs['lstmSize'], dropout=0.5, recurrent_dropout=0.5))
    elif (configs.get('networkType') == "bidirectional"): 
        model.add(Bidirectional(LSTM(configs['lstmSize'], dropout=0.4, recurrent_dropout=0.4)))
    else:
        model.add(LSTM(configs['lstmSize'], dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid')) 
    return(model)   
 
 