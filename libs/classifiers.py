

from libs import utilitylib
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import graphviz, pdb
from libs import evallib
from sklearn.cluster import KMeans
import numpy as np
import os

def decisionTreeClassifier(xTrain,yTrain,xTest,yTest, feMethod= None):
    # feMethod: feature engineering method. it can be word of bag or nothing.
    names = ["Decision Tree", "Random Forest"]
    classifiers = [
                    DecisionTreeClassifier(max_depth=4, criterion = 'entropy'),
                    RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1) ] 
    

    for name, clf in zip(names,classifiers):
        yTrain = yTrain.reshape(yTrain.size)
        clf.fit(xTrain,yTrain)

        if name == "Decision Tree":
            dotData = tree.export_graphviz(clf,out_file = None,
            filled = True, rounded = True,  
            class_names = True)
            graph = graphviz.Source(dotData)
            graph.render(os.path.join('results',feMethod))
        print("feature engineering 's name:",feMethod)    
        evallib.classicationReport(clf,xTest,yTest,"Test data set")
        evallib.classicationReport(clf,xTrain,yTrain, "Train data set")


def cluster( data,  colName, clusters = 10 ):

    data = data.reshape(-1,1) #; print('data: ', data);pdb.set_trace()
    missing = ~np.isfinite(data)
    mu = np.nanmean(data, 0, keepdims=1)

    dataHat = np.where(missing, mu, data)  # filled missing with mean value
    model = KMeans(n_clusters = clusters, init='k-means++', 
            max_iter=100, n_init=1, verbose=0, random_state=3425)
    model.fit(dataHat)
    labels = model.fit_predict(dataHat)
    dataClustered = []
    for _id , label in enumerate(labels):
        if data[_id]: 
            cluster = colName + "_" + str(int(model.cluster_centers_[label][0]))
        else:  # missing data 
            cluster = colName +"_NA"
        dataClustered.append(cluster)
    return dataClustered