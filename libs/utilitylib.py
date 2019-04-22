
import numpy as np 
import matplotlib.pyplot as plt
import random
import string
import pdb,os
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import graphviz 
def getAcc(yPred,yTrue):

    acc = np.sum(yPred == yTrue)/(yPred.size + .001)
    return round(float(acc),3) 

def getPrec(yPred, yTrue):

    prec = sum(yPred * yTrue)/ (sum(yPred) + .001)
    return round(float(prec),3) 

def getRecall (yPred, yTrue):
    recall = sum(yPred * yTrue) / ( sum(yTrue) + .001)
    return round(float(recall),3) 

def roc(yPredProb, yTrue, fileName = None, thresholds = np.arange(.5,.99,.03)):
    ''' we change the threshold and get the new roc point''' 
    temp = []
    roc = {'acc' : [], 'prec':[], 'recall':[], 'thresholds':thresholds}
    for thr in thresholds:
        yPred = (yPredProb >= thr) 
        roc['acc'].append(getAcc(yPred, yTrue))
        roc['prec'].append(getPrec(yPred, yTrue))
        roc['recall'].append(getRecall(yPred, yTrue))

    for key in roc:
        print(key,":",roc[key]) 
    if fileName: 
        plt.figure()
        plt.plot( roc['recall'], roc['prec'],)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.savefig(fileName)
    return roc

def plotHistory(hist, fileName):
    plt.figure()
    plt.subplot(211)
    for key in hist.data.keys():
        if "val" in key: 
            values = hist.data[key]
            plt.plot(values, label = key)
    plt.legend(loc='best', numpoints=1) 
    plt.grid()
    plt.subplot(212)
    # pdb.set_trace()
    for key in hist.data.keys():
        if "val" not in key: 
            values = hist.data[key]
            plt.plot(values, label = key)
    plt.legend(loc='best', numpoints=1)  
    plt.grid()
    plt.savefig(fileName)






def timeDiff(tic,toc):
    diff= round(toc - tic)
    (minDiff, secDiff) = divmod(diff,60)
    (hourDiff,minDiff) = divmod(minDiff,60) 
    ret = 'Time passed: {}hour:{}min:{}sec'.format(hourDiff,minDiff,secDiff)
    print(ret)
    return ret


def printColorActions(_dict, htmlFile): 

    """ 
    color each action (word) of a sequence with the given intensity
    _dict[i]['seqArr'] : list of actions in sequence i .e.g. ['a','c', 'f'] or [1,2,3,53]
    _dict[i]['val'] : list of intensity values for the actions. has to be between zero and one.
    """
    html = generateHTML(_dict)
    with open(htmlFile,'w+') as f:
        f.write(html)
    return 


def readVocab(inputFile):
    vocab = {}
    with open(inputFile,'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        event, _id = line.replace('\n','').split('\t')
        vocab[event] = int(_id) 
    return vocab


def getOutputFiles(configs):
    configStr = 'actions{actionCount}_trains{trainSize}_maxlen{maxlen}_embedding{embeddingSize}_epochs_{epochs}_conv1d_{conv1D}_lstmsize_{lstmSize}'.format(**configs)
    outputFiles = {'hist':os.path.join('results','hist_'+configStr+'.png'),
                    'roc':os.path.join('results','roc_'+configStr+'.png'),
                    'model': { 'weights': os.path.join('models','weight_' + configStr + '.h5'),
                                'json': os.path.join('models','json_model_' + configStr + '.txt'),
                                'mapping': os.path.join('models','mapping_' + configStr + '.tsv'),
                                'configs': os.path.join('models','configs_' + configStr + '.tsv')},
                    'topology' : os.path.join('models','topology_' + configStr + '.jpg')}  
    return outputFiles                 


def logOutput(roc,configs,tic,toc):
    with open(r'performance\performance.txt','a+') as f:
        f.write(SEQUENCE_FILE + "\n")
        
        for key, val in roc.items():
            f.write(key + ":" + ",".join(map(str,val)) + "\n")
        f.write("---------------------------------------- \n")
        for key, val in configs.items():
            f.write(key + ":" + str(val) + "\n")
        f.write("time taken:" + utilitylib.timeDiff(tic,toc) + "\n \n \n") 
        f.write("================================================= \n ")   

def logResultsCV(results,tic,toc):
    with open(r'performance\performance.txt','a+') as f:
        f.write(SEQUENCE_FILE + "\n")
        for key, val in configs.items():
            f.write(key + ":" + str(val) + "\n")
        f.write("---------------------------------------- \n")
        for metric, vals in results.items():
            print("%s: %.2f (%.2f) MSE" % (metric,vals.mean(), vals.std()))
            f.write("%s: %.2f (%.2f) MSE \n" % (metric,vals.mean(), vals.std()))
        f.write("time taken:" + utilitylib.timeDiff(tic,toc) + "\n \n \n") 
        f.write("================================================= \n ")   

def logPerformance(perf):
    with open(r'performance\performance.txt','a+') as f:
        f.write(SEQUENCE_FILE + "\n")
        for key, val in configs.items():
            f.write(key + ":" + str(val) + "\n")
        f.write("---------------------------------------- \n")
        for metric, _dict in perf.items():
            f.write("%s: \n"%metric )
            for embedSize, val in _dict.items():
                f.write("%d:"%embedSize + ",".join(map(str,val)) + "\n")

        f.write("================================================= \n ")   

def logModel(s):
    with open(r'performance\performance.txt','a+') as f:
        print(s, file=f)
