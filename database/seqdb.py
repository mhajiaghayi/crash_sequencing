


import pdb 
import numpy as np 
import pandas as pd
import random,math
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
from libs.classifiers import cluster


class Seqdb: 
    def __init__(self, path, crashIndex= None):
        self.path = path 
        self.map = None 
        self.actionCount = 0
        self.maxSeqSize = 0
        if not crashIndex:  
            self.crashIndex = "1"
        else:
            self.crashIndex = crashIndex


    def preprocess(self, maxSeqSize, dedup, filters, testFlag):

        if not testFlag:
            self.data.loc[:,'systemFreeSpaceMbClustered'] = cluster(self.data.systemFreeSpaceMb, colName = "systemFreeSpaceMb" , clusters = 10)
            self.data.loc[:,'processSpeedMhzClustered'] = cluster(self.data.processSpeedMhz, colName = "processSpeedMhz", clusters = 10)
            self.data.loc[:,'ramMbClustered'] = bucketizedRAM(self.data.ramMb)
            self.data.version.fillna("version_NA")
            sysColumns = ['platform','version','systemFreeSpaceMbClustered','processSpeedMhzClustered','ramMbClustered']
        else:
            sysColumns = []
        self.data.platform.fillna("platform_NA")
        self.data.crashIds.fillna("0")
        seqsProcessed = []
        crashIdsProcessed = []
        for _id, seq in enumerate(self.data.seq):
            if isinstance(seq,str):
                seq = seq.split(',')
            seq = filter(seq,filters)

            if dedup: seq= removeImmediateDuplicate(seq)
            seq = truncate( seq, maxSeqSize )
            if not seq:
                seq = ["no_action"]
            systems = [str(self.data[col][_id]) for col in sysColumns] 
            seqExtended = ",".join(seq + systems)    
            seqsProcessed.append(seqExtended)          
            crashIds = self.data.crashIds[_id]
            crashIdsProcessed.append(int(self.crashIndex in str(crashIds)))
        return( seqsProcessed, crashIdsProcessed )



    def loadData(self,  maxSeqSize = 15 , testSplit = .5, dedup = False,samples = None, filters = None, testFlag = False, p2nRatio= None):
        ''' inputs:
            dedup: if True, it will remove the immediate duplicates in the sequence
        '''
        self.data = pd.DataFrame(pd.read_csv(self.path, sep='\t', names=['session','seq','crashIds','platform','version','model','processSpeedMhz','ramMb','systemFreeSpaceMb']))
        print ('database row count:', self.data.shape[0])
        if samples:
            ids = random.sample(range(self.data.shape[0]), samples)
            self.data = self.data.loc[ids,:]  
       
        self.seqs, self.crashes = self.preprocess(maxSeqSize, dedup, filters, testFlag) 
        seqIds = self.tokenize(self.seqs)   
        # (xTrain, yTrain), (xTest, yTest) = self.splitData(seqIds, self.crashes, testSplit)
        (xTrain, yTrain), (xTest, yTest) = self.splitData(seqIds, self.crashes, testSplit, p2nRatio)
        return ([(xTrain,yTrain), (xTest,yTest)])


    def loadBagOfWordsData(self, testSplit = .5):
        # Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.  
        vectorizer = CountVectorizer(tokenizer= MyTokenizer())
        seqsBoW = vectorizer.fit_transform(self.seqs).toarray()  
        (xTrain, yTrain), (xTest, yTest) = self.splitData(seqsBoW, self.crashes, testSplit,p2nRatio=1)                            
        return ([(xTrain, yTrain), (xTest, yTest)])


    def tokenize (self, seqs):
        actionIdMap = {}
        arrs = []
        actionCount = 0
        longestSeqSize = 0 
        for seq in seqs:
            if isinstance(seq, str):
                seq = seq.split(',')
            seqSize = len(seq)
            arr = seqSize*[0]
            for i, action in enumerate(seq): 
                if action in actionIdMap:
                    arr[i] = actionIdMap[action]
                else:
                    actionCount = actionCount + 1 
                    actionIdMap[action] = actionCount
                    arr[i] = actionCount
            arrs.append(arr)
            if seqSize > longestSeqSize: longestSeqSize = seqSize 
        self.map = actionIdMap
        self.actionCount = actionCount
        self.longestSeqSize = longestSeqSize
        #print("actionIdMap: ", actionIdMap);pdb.set_trace()
        return (arrs)


    def getActionMap(self):
        return(self.map)


    def saveActionMetaData(self, fileName):
        row = 1 
        with open(fileName, 'w') as f:
            f.write("Word\tFrequency\n")
            for action, _id in self.map.items():
                if row < len(self.map.keys()):
                    f.write(action + "\t" +str(_id) + "\n")
                    row = row + 1 
                else:
                    f.write(action + "\t" +str(_id))
        return

    def splitData(self, seqIds, labels, testSplit, p2nRatio=1):


        dataSize = len(seqIds)
        print('Number of crashes in unbalanced data: ', sum(labels))
        print('Total number of sequences in unbalanced data: ', len(seqIds))  
        random.seed(3)
        trainIds = random.sample(range(dataSize), round(dataSize * (1-testSplit)))
        testIds = [x for x in range(dataSize) if x not in trainIds]
        xTest = []
        yTest = []
        xTrain = []
        yTrain = []
        for id in testIds:
            xTest.append(seqIds[id])
            yTest.append(labels[id])
        for id in trainIds:
            xTrain.append(seqIds[id])
            yTrain.append(labels[id])

        xTrain,yTrain,xTest,yTest = self.shuffle(xTrain,yTrain,xTest,yTest)
        return([(np.array(xTrain),np.array(yTrain, dtype = np.int32).reshape((-1,1)) ), (np.array(xTest),np.array(yTest,dtype=np.int32).reshape((-1,1)))])
    
    def shuffle(self,xTrain,yTrain,xTest,yTest):
        np.random.seed(2)
        testId = np.random.permutation(len(xTest))
        trainId= np.random.permutation(len(xTrain))
        xTrain = [xTrain[i] for i in trainId]
        yTrain = [yTrain[i] for i in trainId]
        xTest = [xTest[i] for i in testId]
        yTest = [yTest[i] for i in testId]
        return([xTrain,yTrain,xTest,yTest])
    def summary(self):
        setSeq = {}
        for _id, seq in enumerate(self.seqs):
            if str(seq) in setSeq:
                setSeq[seq] += 1 
            else:
                setSeq[seq] = 1 

        print("Total number of sessions in database:{0}, Total number of crashes in database: {1}".format(len(self.seqs), sum(self.crashes)))
        values = [setSeq[_str] for _str in setSeq]
        print("top 01 perc of sequences appeared more than %f)"%(np.percentile(values,99)))
        print("top 05 perc of sequences appeared more than %f)"%(np.percentile(values,95)))
        self.plotSystemHistogram(self.data)

    @staticmethod
    def plotSystemHistogram(data, histFileName= 'results/system_hist.png', statFileName = 'results/system_statSummary.txt' ):
        columns = ['processSpeedMhz', 'ramMb', 'systemFreeSpaceMb']
        plt.figure()
        with open(statFileName, 'w+') as statFile: 
            for i, col in enumerate(columns):
                plt.subplot(int("31"+ str(i+1)))
                plt.hist(data[col].dropna(), bins = 'auto')
                plt.title("histogram for %s"%col)
                plt.tight_layout()
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                statFile.write('stat summary for : %s'%col)
                stat = data[col].describe()
                statFile.write(stat.to_string())
                naRatio = np.count_nonzero(np.isnan(data[col])) * 1.0/ len(data[col])
                statFile.write(' \n NA ratio for : %s is %f, total records %d'%(col,naRatio,len(data[col])))
                statFile.write('\n \n ')
        plt.savefig(histFileName)

class MyTokenizer(object):
    def __call__(self, s):
        return s.split(',')


def removeImmediateDuplicate(seq): 
    ''' This function remove the immeidate dupplicates of the array input data
    e.g. if row = c(1,1,3,4,4,6), the output is rowDedup = c(1,3,4,6) ''' 
    dedup = list()
    dedup.append(seq[0])
    if len(seq) > 1:
        for i in range(1, len(seq)):
            if seq[i] != seq[i-1]:
                dedup.append(seq[i])
    return dedup


def filter(seq, filters = None):
    if not filters:
        return seq
    clean = [action for action in seq if action not in filters]
    return clean


def truncate(seq, maxSeqSize):
    if not seq:
        return seq
    start = max(len(seq)-maxSeqSize,0)
    return(seq[start:len(seq)])


def bucketizedRAM(rams):
    buckets = []
    for ram in rams:
        if not math.isnan(ram): 
            bucket = "ramMb_" + str(int(ram))
        else:
            bucket = "ramMb_NA"
        buckets.append(bucket)
    return( buckets)
