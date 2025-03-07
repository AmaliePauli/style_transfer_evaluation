import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr,spearmanr,kendalltau
from itertools import permutations
from numpy import random
import itertools
from compare import output_on_dataset,output_on_dataset_test,output_on_dataset_refmethod
from data import *
import random

# method sepcific import
from QuestEval.questeval.questeval_metric import QuestEval

def output_lai(method):
    folder = 'output'
    column_i='input'
    column_o='output'

    data = Lai3()
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    
###################
class QEval:
    def __init__(self,name:str='questEval'):
        self.name = name
        self.questeval = QuestEval()

    def predict(self,df,column_i:str='input',column_o:str='output'):
        score = self.questeval.corpus_questeval(
                hypothesis=list(df['output']), 
                sources=list(df['input'])
                )

        return score['ex_level_scores']
    
method=QEval()

output_on_dataset(method)
output_on_dataset_test(method)
output_on_dataset_refmethod(method)
print('done')
#################################

