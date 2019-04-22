import pandas as pd
import pdb
class ReportSummary: 

    def __init__(self,vocab,isEventEncoded):
        self.contributorsList = {}
        self.blockersList = {} 
        self.idMap = {}
        for event, _id in vocab.items():
            if isEventEncoded:
                self.idMap[_id] = event
            else:
                self.idMap[event] = event 
        self.idMap[0] = "0"
        self.idMap["0"] = "0"


    def add(self,contributors, blockers):

        try:
            contributorsStr = "|".join(map(str,[self.idMap.get(_id) for _id in contributors]))
        except:
            pdb.set_trace()

        if contributorsStr: 
            if contributorsStr in self.contributorsList:
                self.contributorsList[contributorsStr] +=1   
            else:
                self.contributorsList[contributorsStr] = 1 


        blockersStr = "|".join(map(str,[self.idMap[_id] for _id in blockers]))
        if blockersStr:
            if blockersStr in self.blockersList:
                self.blockersList[blockersStr] += 1   
            else:
                self.blockersList[blockersStr] = 1

    def sort(self, topK = None):
        self.contributorsDf = pd.DataFrame(sorted(self.contributorsList.items(), key=lambda x:x[1],reverse = True)[0:topK], columns = ['Contributors', "Frequency"])
        self.blockersDf = pd.DataFrame(sorted(self.blockersList.items(), key=lambda x:x[1],reverse = True)[0:topK],columns = ['Blockers', "Frequency"])
        


    def save(self,fileName):

        try:
            df = pd.concat([self.contributorsDf, self.blockersDf],axis =1)
            df.to_csv(fileName, sep=',', encoding='utf-8')
        except:
            print("ERROR: Couldn't create csv file most likely because the file is opened!!!!")

    def saveHTML(self,seqs,htmlFile,showEncoded = False):

        html = '''
                <!DOCTYPE html>
                <html>
                <head>
                <style>
                table {
                    font-family: arial, sans-serif;
                    border-collapse: collapse;
                    width: 100%;
                }

                td, th {
                    border: 1px solid #dddddd;
                    text-align: left;
                    padding: 8px;
                }

                tr:nth-child(even) {
                    background-color: #dddddd;
                }
                </style>
                </head>
                <body>

                <table> 
                <tr> 
                    <th> id </th>
                    <th> Sequence </th>
                    <th> Prediction </th> 
                    <th> True Label </th> 
                    <th> Confidence Score </th>
                <tr>  
        '''
        for seq in seqs:
            colorCodedSequence = self.getHtmlColorCodedSequence(seq,showEncoded)
            html = html +  '''
                <tr> 
                    <td> {id} </td>
                    <td> {sequence} </td> 
                    <td> {pred} </td> 
                    <td> {trueLabel} </td>
                    <td> {confidence} </td>
                </tr>  
                '''.format(**{  'id': seq.id,
                                'sequence': colorCodedSequence,
                                'pred': seq.pred,
                                'trueLabel':seq.label,
                                'confidence' : seq.conf,
                             }
                           )

        html = html + '''
                        </table>
                        </body>
                        </html>'''
       
        with open(htmlFile,'w+') as f:
            f.write(html)
        return 
    def getHtmlColorCodedSequence(self,seq,showEncoded):
        ret = ""
        for i, event in enumerate(seq.arr):
            if not showEncoded:
                event = self.idMap[event] 
            if i in seq.blockerIds:
                eventColorCoded =  '<span style="background-color:rgba(255,0,0,{0})"> {1} </span>'.format(seq.eventsEffect[i],event)
            elif i in seq.contributorIds:
                eventColorCoded = '<span style="background-color:rgba(0,255,0,{0})"> {1} </span>'.format(seq.eventsEffect[i],event)
            else:
                eventColorCoded = '<span style="background-color:rgba(255,0,0,0)"> {0} </span>'.format(event)
            ret = ret + eventColorCoded
        return ret