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
from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction

import bert_score
import evaluate

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
class SelfBleu:
    def __init__(self,name:str='SelfBleu'):
        self.smooth = SmoothingFunction()
        self.name = name
      

    def predict(self,df,column_i:str='input',column_o:str='output'):
        return list(df.apply(lambda x: sentence_bleu([x[column_o]],x[column_i],smoothing_function=self.smooth.method1), axis=1 ))
    
method=SelfBleu()
output_on_dataset(method)
print(method.name)
output_on_dataset_test(method)
output_on_dataset_refmethod(method)
print('done')
#################################


class SelfBert_score:
    def __init__(self,name:str='SelfBert_score', model_type:str='microsoft/deberta-xlarge-mnli'):
        self.name = name
        self.model_type = model_type

    def predict(self,df,column_i:str='input',column_o:str='output'):
        
        P, R, F1 = bert_score.score(list(df[column_i]), list(df[column_o]), model_type=self.model_type, verbose=False)
        
        return F1
    
    
method=SelfBert_score()
output_on_dataset(method)
print(method.name)
output_on_dataset_test(method)
output_on_dataset_refmethod(method)
print('done')
###################

class SelfMeteor:
    # for large r model evaluate.load('bleurt', 'bleurt-large-512')
    def __init__(self,name:str='SelfMeteor',column_i='input',column_o='output'):
        self.meteor = evaluate.load('meteor')
        self.name = name
        self.column_i=column_i
        self.column_o=column_o
    def predict(self,df,column_i:str='input',column_o:str='output'):
        
        return list(df.apply(lambda x: self.meteor.compute(predictions=[x[column_o]], references=[x[column_i]])['meteor'], axis=1 ))
    
method=SelfMeteor()
output_on_dataset(method)
print(method.name)
output_on_dataset_test(method)
output_on_dataset_refmethod(method)
print('done')
######################