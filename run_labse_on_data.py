
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
from sentence_transformers import SentenceTransformer,util
class LabseCos:
    def __init__(self,name:str='SelfLaBSECos'):
        self.name = name
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        

    def predict(self,df,column_i:str='input',column_o:str='output'):
        
        embedding_1= self.model.encode(list(df[column_i]), convert_to_tensor=True)
        embedding_2 = self.model.encode(list(df[column_o]), convert_to_tensor=True)
        
        scores=util.pytorch_cos_sim(embedding_1, embedding_2).cpu().numpy()
        scores = np.diagonal(scores)
        return scores
    
method=LabseCos()

###
output_on_dataset(method)

output_on_dataset_test(method)
output_on_dataset_refmethod(method)
print('done')
#################################

