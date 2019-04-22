
from libs import utilitylib
class Logging:
    def __init__(self, seqFileName, perfFileName=r'performance\performance.txt'):
        self.seqFileName = seqFileName 
        self.perfFileName = perfFileName


    def finalOutput(self,roc, configs, tic, toc):
        with open(self.perfFileName,'a+') as f:
            f.write(self.seqFileName + "\n")
            for key, val in roc.items():
                f.write(key + ":" + ",".join(map(str,val)) + "\n")
            f.write("---------------------------------------- \n")
            for key, val in configs.items():
                f.write(key + ":" + str(val) + "\n")
            f.write("time taken:" + utilitylib.timeDiff(tic,toc) + "\n \n \n") 
            f.write("================================================= \n ")   

    def cvResults(self,results,tic,toc):

        with open(self.perfFileName,'a+') as f:
            f.write(self.seqFileName + "\n")
            for key, val in configs.items():
                f.write(key + ":" + str(val) + "\n")
            f.write("---------------------------------------- \n")
            for metric, vals in results.items():
                print("%s: %.2f (%.2f) MSE" % (metric,vals.mean(), vals.std()))
                f.write("%s: %.2f (%.2f) MSE \n" % (metric,vals.mean(), vals.std()))
            f.write("time taken:" + utilitylib.timeDiff(tic,toc) + "\n \n \n") 
            f.write("================================================= \n ")   

    def performance(self,perf):
        with open(self.perfFileName,'a+') as f:
            f.write(self.seqFileName + "\n")
            for key, val in configs.items():
                f.write(key + ":" + str(val) + "\n")
            f.write("---------------------------------------- \n")
            for metric, _dict in perf.items():
                f.write("%s: \n"%metric )
                for embedSize, val in _dict.items():
                    f.write("%d:"%embedSize + ",".join(map(str,val)) + "\n")

            f.write("================================================= \n ")   

    def model(self,s):
        with open(self.perfFileName,'a+') as f:
            print(s, file=f)