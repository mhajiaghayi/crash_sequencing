

import string
import random
import numpy as np
import os 
import pdb

class PsudoSeqdb:
	
	''' we create a psudo seq detabase. '''
	def __init__( self, dbSize, actionCount, maxSeqSize, equalSize = False ):
		self.dbSize = dbSize ; 
		self.actionCount = actionCount 
		self.maxSeqSize = maxSeqSize
		self.seqs = list()
		self.crashes = list()
		self.equalSize = equalSize


	def generate( self, crashContributors, crashBlockers, hardwareOptions = 1, hardwareAssociates = [], p2nRatio = .5):
		''' 
		 generate random sequence and save it to db if it doesn't already exits. Also the ratio of crash vs non crash 
		 must prevail.
		 crash logic: For any sequence generate if it constitutes all crashAssociate actions and not any action from 
		 notCrashAssociate, it leads to the crash we see a crash. 

		 iputs: 
		 	crashAssociates: list of "actions" that IF the followed by a specific hardware architecture will lead to 
		 	a crash. 
		 	hardwareAssociates: list of "hardware" items that if comes after crashAssociates will lead to a crash.
		 	hardwareOptions: number of hardware options (Ex. 5 different types of RAM shown by 1, 2, ..., 5).
		 	nonCrashAssociate: list of actions that any of them prevents a crash
		 	p2nRatio: positve(crashes) to negative (non crashes) ratio.
		'''
		actions = string.ascii_lowercase[ 0:self.actionCount ]
		pCount = 0                        # crashes (positive)
		nCount = 0 	                      # no crashes (negative)
		seqSet = set() 
		actionProb = self.getActionProb( actions, crashContributors )
		for item in hardwareAssociates:
			crashContributors.append(item)

		while len(self.seqs) < self.dbSize: 
			seqSize = self.maxSeqSize
			if not self.equalSize: seqSize = random.choice( range( 1, self.maxSeqSize ) )

			seq = self.createRandomSeq(seqSize, actions, actionProb, hardwareOptions)
			if str(seq) in seqSet:    # This seq already exists 
				continue 
			toBeAdded = False

			if self.doesCrash(seq, crashContributors, crashBlockers):
				# print("Crash Happened")
				if pCount <= self.dbSize * p2nRatio:
					pCount = pCount + 1 
					crash  = 1 
					toBeAdded = True
			else: 
				# print("Crash didn't happen")
				if nCount <= self.dbSize * (1-p2nRatio):
					nCount = nCount + 1
					toBeAdded = True 
					crash = 0

			if toBeAdded:
				self.seqs.append(seq)
				self.crashes.append(crash)
				seqSet.add(str(seq))
		print("Number of Positive Sequences: ", pCount)		
		print("Number of Negative Sequences: ", nCount)		


	def save( self, outputFile ):
		with open(outputFile, 'w') as f:
			for i, seq in enumerate(self.seqs):
				# hardware = seq[-1]
				# seq = seq[:-1]
				f.write(str(i) + "\t" +','.join(seq) + "\t" + str(self.crashes[i])) 
				if i < len(self.seqs)-1:
					f.write("\n")


	def getActionProb( self, actions, crashContributors ):
		assocCount = len(crashContributors) 
		freqFactor = 3  # crashContributors will appear 3 times more than other actions 
		prob = np.array(len(actions) *[1])/(assocCount*freqFactor + len(actions)-assocCount) 
		assocIds = self.getAssocActionIndex(actions,crashContributors)
		prob[assocIds] = freqFactor*prob[assocIds]
		return prob


	def getAssocActionIndex( self, actions, crashContributors ):
		ids = []
		_map = {}
		for i,action in enumerate(list(actions)):
			_map[action] = i

		for action in crashContributors:
			ids.append(_map[action])
		return ids


	def createRandomSeq( self, seqSize, actions, actionsProb, hardwareOptions ):
		seq = np.random.choice(list(actions), seqSize, p=actionsProb).tolist()
		if hardwareOptions:
			hardware = random.randint( 1, hardwareOptions )
			seq.append(str(hardware))
		return seq


	def doesCrash( self, seq, contributor, blocker ):
		'''  Decides for given seq and a list of crash associate actions, hardware crash associates 
		and not-crash associate actions whether we see a crash or not.
		Logic: 
		''' 
		# initial inspection 
		if len(seq) < len(contributor):
			return False

		for action in blocker:
			if action in seq:
				return False 

		ptrSeq = 0 
		ptr = 0 
		notFound = False


		# We have two pointers on seq and contributor and move them to see if we can cover all 
		# the actions of crashAssocAction in the seq. if seq contains all the actions in contributor
		# in the given order, then we return true. 
		while ptrSeq < len(seq) and  ptr < len(contributor):
			while seq[ptrSeq] != contributor[ptr]:
				ptrSeq = ptrSeq + 1
				if ptrSeq == len(seq):
					notFound = True
					break 
			ptr = ptr + 1  
		isCovered = (ptr == len(contributor))

		if not notFound and isCovered:
			return True 
		else:
			return False ; 






parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEQUENCE_FILE  = os.path.join(parentdir,"data/test_action_20_maxSeq15_10K.txt")

pseqdb =  PsudoSeqdb(maxSeqSize = 15,
                    actionCount = 20,
                    dbSize = 10000,
                    equalSize = True)

pseqdb.generate(crashContributors = ['f','b','c'],
                crashBlockers = ['e'],
                p2nRatio = .5,
                hardwareOptions = None,
                hardwareAssociates = [])

pseqdb.save(outputFile = SEQUENCE_FILE)