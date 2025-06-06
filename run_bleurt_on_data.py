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
# method sepcific import
#import evaluate 
#https://huggingface.co/Elron/bleurt-large-512
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SelfBleurt:
    def __init__(self,name:str='SelfBleurt'):
        self.tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-large-512")
        self.model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-large-512")
        self.model.eval()
        self.name = name
        
    def predict(self,df,column_i:str='input',column_o:str='output'):
        inputs = list(df[column_i])
        outputs= list(df[column_o])
        
        with torch.no_grad():
              scores = self.model(**self.tokenizer(inputs, outputs, return_tensors='pt',padding=True, truncation=True))[0].squeeze()

    
        print()
        print(np.isnan(scores).sum())
        return scores 

method=SelfBleurt()
    

output_on_dataset(method)
output_on_dataset_test(method)
output_on_dataset_refmethod(method)

print('done')
#################################

