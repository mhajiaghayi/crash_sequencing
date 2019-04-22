# evaluation lib
from libs import utilitylib
from sklearn.metrics import classification_report

def evaluate(xTrain,yTrain,model,batchSize):
    # yTrain = yTrain.reshape((yTrain.size,1))
    yTrainPred = model.predict_classes(xTrain, verbose = False)
    # pdb.set_trace()
    print('\n model.predict:')
    print('loss:',0, ' acc:',utilitylib.getAcc(yTrainPred,yTrain), ' recall:',utilitylib.getRecall(yTrainPred,yTrain), ' prec:', utilitylib.getPrec(yTrainPred,yTrain))


def sanityCheck(xTrain,yTrain,model,hist,batchSize):
    # yTrain = yTrain.reshape((yTrain.size,1))

    print ('sanity check starts ............')
    loss,acc,recall,prec = model.evaluate(xTrain, yTrain,
                            batch_size=batchSize)  
    print('\n model.evaluate:')
    print('loss:',loss, ' acc:',acc, ' recall:',recall, ' prec:', prec)

    yTrainPred = model.predict_classes(xTrain, verbose = False)
    # pdb.set_trace()
    print('\n model.predict:')
    print('loss:',0, ' acc:',utilitylib.getAcc(yTrainPred,yTrain), ' recall:',utilitylib.getRecall(yTrainPred,yTrain), ' prec:', utilitylib.getPrec(yTrainPred,yTrain))

    print( ' \n model.fit')
    perf = hist.history 
    print(' last output')
    print('loss:',perf['loss'][-1], ' acc:',perf['acc'][-1], ' recall:',perf['recall'][-1], ' prec:' ,perf['precision'][-1])

    print('minimum lost:', min(perf['loss']))



def evaluate2(xTest,yTest,model,batchSize):
    loss,acc,recall,prec = model.evaluate(xTest, yTest,
                            batch_size=batchSize)  

    # yPred = model.predict_classes(xTest, batch_size = batchSize)
    # yPredTrain = model.predict_classes(xTrain, batch_size = batchSize)
    yPredProb = model.predict(xTest,batch_size = batchSize)
    print('Test loss:{0}, accuracy:{1}, recall:{2}, prec:{3}'.format(loss,acc,recall,prec))


def classicationReport(clf,xData,yData,reportName):
    yPred = clf.predict(xData)
    print(" classifcation report for %s ....",reportName)
    print(classification_report(yData,yPred))
    score = clf.score(xData,yData)
    print("accuracy:",score)  