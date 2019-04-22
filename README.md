
By: Mahdi Hajiaghayi, Ehsan Vahedi
Date : April 2018



## Intro: 
this is the implementation of our work on crash prediction and extraction based on LSTM network
it consists of three main modules. 
1) seqdb_lstm_train: it trains a network based on the given sequential data. the format of the input data is described below
2) seqdb_lstm_test: for given sequence or input data, it predicts whether crash (1) occurs or not (0)
3) seqdb_lstm_extract: using the network trained in step 1, it extracts the crash contributors and blockers and save the 
aggregation results in csv file and also return an html that has highlighted contributors and blockers in the given file. 



## Install
The requirement packages are listed in requirement.txt. it's recommended to use Anaconda  



## Usage
In order to train a network for your input data, you first need to specify the path of your data in 
_init_global_vars_train. The input data must have three columns at least: id, sequence, output. For more information please look at data/test_action.txt. In this file, we have one extra column as well that was supposed to specify the platform.
this file is processed by database/seqDb.py class. Hence, by modifying this file you can have more columns.

As the optimal hyper parameters might be different file inputs, we have included another file 
called seqdb_lstm_baysian_opt.py that basically optimizes the hyper parameters and then you can use them
in the seqdb_lstm_train.py.

You can kick in the tra
ng by 
```
python seqdb_lstm_train.py 
```

the outputs of the training is stored in /model. 
for extraction, you need to specify the postfix of the model file in the init_global_vars_extract.py and just 
run the 

```
python seqdb_lstm_extract.py 
```

