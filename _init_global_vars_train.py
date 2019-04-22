
SEQUENCE_FILE = "data\\test_action_20_maxSeq15_20K.txt"
CRASH_INDEX = "1"
# SEQUENCE_FILE = "data\\activityseq_balanced_2018-02-05_2018-02-09_Excel_apphangb1_Production.tsv"
# SEQUENCE_FILE = r"C:\Users\mahajiag\Documents\tmp\crash\eventseq_balanced_2018-02-05_2018-02-09_Excel_apphangb1_Production.tsv"

ROOT_DIR = 'tfboard'

ACTIONS_TO_BE_FILTERED = ['']


# configs = { 'batchSize' : 512,
#             'maxSeqSize': 40,
#             'testSplit': .5,
#             'embeddingSize': None,
#             'epochs': 400,
#             'dedup': 0,
#             'conv1D':0,
#             'lstmSize': None,
#             'testFlag': False,
#             'p2nRatio':1,
# 		    'networkType':"bidirectional"}    

configs = { 'batchSize' : 512,
            'maxSeqSize': 17,
            'testSplit': .5,
            'embeddingSize': None,
            'epochs': 250,
            'dedup': 1,
            'conv1D':0,
            'lstmSize': None,
            'testFlag': True,
            'p2nRatio':1,
            'networkType':"bidirectional"
            }    


classWeights = {0: .5,
		1: .5}

