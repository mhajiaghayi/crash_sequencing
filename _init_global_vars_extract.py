

CRASH_INDEX = "1"
ACTIONS_TO_BE_FILTERED = ['Office.Excel.Command.CloseAll','Office.Text.GDIAssistant.FontManagerDestructor','Office.UX.RibbonX.RibbonIsBeingCleanedUp']  # 'Office.Text.GDIAssistant.FontManagerDestructor'



SEQUENCE_FILE  = "data\\test_action_20_maxSeq15_10K.txt"
# SEQUENCE_FILE = "data\\ActivitySeq_2017-10-01_2017-10-10_Excel_apphangb1_Insiders.tsv"
# SEQUENCE_FILE  = r'C:\Users\mahajiag\Documents\tmp\crash\activityseq_balanced_2018-02-05_2018-02-09_Excel_apphangb1_Production.tsv'


# MODEL_SPECIFIC = r'actions1211_trains49896_maxlen45_embedding2_epochs_400_conv1d_0_lstmsize_6'
MODEL_SPECIFIC = r'actions20_trains5000_maxlen15_embedding3_epochs_250_conv1d_0_lstmsize_6'
CONFIG_FILE = "models\configs_%s.tsv"%(MODEL_SPECIFIC) 
# CONFIG_FILE = "models\configs_actions17_trains10000_maxlen16_embedding3_epochs_100_conv1d_0_lstmsize_5.tsv"

# CONFIG_FILE = "models\configs_actions871_trains20000_maxlen40_embedding40_epochs_1000_conv1d_0.tsv"




HTML_OUTPUT_FILE = r'results\color_actions\%s.html'%(MODEL_SPECIFIC) 
# HTML_OUTPUT_FILE= r"results\color_actions\hist_actions17_trains10000_maxlen16_embedding3_epochs_100_conv1d_0_lstmsize_5.html"

REPORT_OUTPUT_FILE = r'results\reports_%s.csv'%(MODEL_SPECIFIC) 