import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr,spearmanr,kendalltau
from itertools import permutations
from numpy import random
import itertools
from compare import output_on_dataset_both,compare_on_synethic,compare_on_synethic_dev,compare_on_repeat_dev,calculate,compare_on_synethic2,output_on_dataset_test_both,output_on_dataset_both
from data import *
import random

# method sepcific import
import transformers
from transformers import pipeline
import torch
from tqdm import tqdm
import datasets
from datasets import Dataset
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
from transformers.pipelines.pt_utils import KeyDataset
dim='c'
target='mean'
samplesize=None
results_sum=pd.DataFrame()



class Prompt8B:
    def __init__(self,name:str='PromptLlama8b'):
        model_id = "meta-llama/Llama-3.1-8B-Instruct"
        self.pipeline = pipeline("text-generation", model=model_id, temperature=1, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
        self.name = name
        
    def predict(self,df,column_i:str='input',column_o:str='output'):
        #inputs = list(df[column_i])
        #outputs= list(df[column_o])
        df['style_to']=df['style_to'].apply(lambda x: 'understandable for layman' if x=='layman' else x)
        df['style_to']=df['style_to'].apply(lambda x: 'addressed to an expert' if x=='expert' else x)
        df['style_to']=df['style_to'].apply(lambda x: 'simple' if x=='simplicity' else x)
        df['style_to']=df['style_to'].apply(lambda x: 'simple' if x=='simplified' else x)
        
        prompt="Evaluate the following completion of a task where a 'source sentence' has been rewritten to be more {} in the style, denoted 'target sentence', Ideally the context and content in the sentence which does not relate to the style should be preserved. Please evaluate on a Likert scale from 1-5 with 5 being the best: 1) how well the meaning is preserved and 2) how well the the style is changed. Return in JSON format with the keys 'meaning' , 'style'. Given the 'source sentence': {} 'target sentence': {}"

        #messages = [
        #    {"role": "system", "content": "You are good at evaluating style and attribute transfer in text"},
        #    {"role": "user", "content": prompt }]
        
        df['prompt'] = df.apply(lambda x: prompt.format(x['style_to'],x['style_to'],x[column_i],x[column_o]), axis=1)
        df['message'] = df['prompt'].apply(lambda x: [{"role": "system", "content": "You are a helpfull assistant"},{"role": "user", "content": x }])
        
        ds = Dataset.from_pandas(df)
        
        # make predictions
        def is_int_or_convertible(value):
            if pd.isna(value):
                return False
            if isinstance(value, int):  
                return True
            if isinstance(value, str): 
                try:
                    eval(value)
                except:
                    return False
                else:
                    return True

                return False 

        preds = []
        for out in tqdm(self.pipeline(KeyDataset(ds, "message"),max_new_tokens=50,pad_token_id=self.pipeline.tokenizer.eos_token_id)):
            preds.append(out[0]['generated_text'][2]['content'])
        #print(preds[0:2])
        #post-processes
        df['outputs'] = preds
        path=os.path.join('prompt_final','outputs_{}_{}.csv'.format(method.name,data.name))
        df.to_csv(path, index=False)
        df['completeC'] = df.outputs.apply(lambda x: 1 if '"meaning"' in x else 0)
        df['pc-1']=df.apply(lambda x: x['outputs'].split('"meaning":')[1].split("}")[0].replace('"','').split(',')[0] if x['completeC']==1 else np.nan,axis=1)
        df['completeC']=df.apply(lambda x: 1 if is_int_or_convertible(x['pc-1']) else 0, axis=1) 
        
        
        #style
        df['completeS'] = df.outputs.apply(lambda x: 1 if '"style"' in x else 0)
        df['ps-1']=df.apply(lambda x: x['outputs'].split('"style":')[1].split("}")[0].replace('"','').split(',')[0] if x['completeS']==1 else np.nan,axis=1)
        df['completeS']=df.apply(lambda x: 1 if is_int_or_convertible(x['ps-1']) else 0, axis=1) 
        
        # replace NONE with mean
        c1=df[df['completeC']==1]['pc-1'].map(float).mean()
        s1=df[df['completeS']==1]['ps-1'].map(float).mean()
        df['pc-1']=df.apply(lambda x: c1 if x['completeC']==0 else x['pc-1'],axis=1)
        df['ps-1']=df.apply(lambda x: s1 if x['completeS']==0 else x['ps-1'],axis=1)    
        df['pc-1'] = df['pc-1'].map(float)
        df['ps-1'] = df['ps-1'].map(float)
        return list(df['pc-1']),list(df['ps-1']),[]

method=Prompt8B()

####
output_on_dataset_test_both(method)
