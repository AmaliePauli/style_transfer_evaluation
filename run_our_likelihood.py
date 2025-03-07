import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr,spearmanr,kendalltau
from itertools import permutations
from numpy import random
import itertools
from compare import output_on_dataset_both,output_on_dataset_test_both
from data import *
import random
# method sepcific import
from transformers import PreTrainedTokenizerFast
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import numpy as np


cuda='cuda:0'
dim='c'
target='mean'
samplesize=None
model_id="meta-llama/Llama-3.1-8B-Instruct"
#model_id="meta-llama/Llama-3.2-3B-Instruct"
results_sum=pd.DataFrame()

###################
class LLMProp:
    def __init__(self,name:str='Our_lkelihood_8b'): 
        self.name = name
        self.model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(cuda)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id)
        self.model.eval()     
        
    def likelihoods(self,logits, force_decode_indices):
        probs = F.softmax(logits, dim=-1)
        probs_force_decode = probs.gather(-1, force_decode_indices.unsqueeze(-1)).squeeze()
        assert probs_force_decode.shape == force_decode_indices.squeeze().shape
        return probs_force_decode 
    
    def calculate_delta(self,logits,start_idx,input_ids,logits2,start_idx2,input_ids2,
                       logits3,start_idx3,input_ids3):       
        first_probs = self.likelihoods(logits[0][start_idx:-1], input_ids[0][start_idx + 1:])
        second_probs = self.likelihoods(logits2[0][start_idx2:-1], input_ids2[0][start_idx2 + 1:])
        third_probs = self.likelihoods(logits3[0][start_idx3:-1], input_ids3[0][start_idx3 + 1:])


        first_loss =np.array(first_probs.tolist()) 
        second_loss =np.array(second_probs.tolist()) 
        third_loss =np.array(third_probs.tolist()) 

        content = np.maximum(first_loss,third_loss)
        delta = second_loss-content 
        s = np.mean(delta)     
        
        c = np.mean(np.log(np.maximum(np.maximum(first_loss,second_loss),third_loss))) 

        u=np.mean(np.log(second_loss))
   

        return pd.Series([c,s,u])
    
    def get_predictions(self,out1,source,style):

        systemprompt="You can repeat sentences, paraphrase sentences or rewrite sentences to change the style or certain attribute of the text while preserving non-related content and context. Your answers contain just the rewrite."
        
        
        template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>{}<|eot_id|><|start_header_id|>user<|end_header_id|>{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
         
        inputprompt1= 'Paraphrase the following sentence: {}'.format(source)
        inputprompt2= 'Rewrite the following sentence to be {}: {}'.format(style,source)
        inputprompt3= 'Repeat the following sentence: {}'.format(source)
        
        inst1=template.format(systemprompt,inputprompt1)
        inst2=template.format(systemprompt,inputprompt2)
        inst3=template.format(systemprompt,inputprompt3)
        output=out1 +'<|eot_id|>'

        #tokenize
        inst1_tok=self.tokenizer.tokenize(inst1)
        inst2_tok=self.tokenizer.tokenize(inst2)
        inst3_tok=self.tokenizer.tokenize(inst3)
        output_tok=self.tokenizer.tokenize(output)
        
        #get indexes
        inst1_ids = self.tokenizer.convert_tokens_to_ids(inst1_tok)
        inst2_ids = self.tokenizer.convert_tokens_to_ids(inst2_tok)
        inst3_ids = self.tokenizer.convert_tokens_to_ids(inst3_tok)
        output_ids = self.tokenizer.convert_tokens_to_ids(output_tok)
        
        # creat inputs
        input1_ids = torch.tensor([inst1_ids+output_ids]).to(cuda)
        start1_idx = len(inst1_ids)   
        attention1_mask = torch.tensor([[1] * len(input1_ids)]).to(cuda)
        
        input2_ids = torch.tensor([inst2_ids+output_ids]).to(cuda)
        start2_idx = len(inst2_ids)   
        attention2_mask = torch.tensor([[1] * len(input2_ids)]).to(cuda)
        
        input3_ids = torch.tensor([inst3_ids+output_ids]).to(cuda)
        start3_idx = len(inst3_ids)   
        attention3_mask = torch.tensor([[1] * len(input3_ids)]).to(cuda)
        
        # get logits
        with torch.no_grad():
            input1_logits = self.model(input_ids=input1_ids, attention_mask=attention1_mask,return_dict=True).logits
            input2_logits = self.model(input_ids=input2_ids, attention_mask=attention2_mask,return_dict=True).logits
            input3_logits = self.model(input_ids=input3_ids, attention_mask=attention3_mask,return_dict=True).logits


        return self.calculate_delta(input1_logits,start1_idx,input1_ids,
                                    input2_logits,start2_idx,input2_ids,
                                   input3_logits,start3_idx,input3_ids)
                           
   
     
    def predict(self,df,column_i:str='input',column_o:str='output'):
        df['style_to']=df['style_to'].apply(lambda x: 'understandable for layman' if x=='layman' else x)
        df['style_to']=df['style_to'].apply(lambda x: 'addressed to an expert' if x=='expert' else x)
        df['style_to']=df['style_to'].apply(lambda x: 'simplified' if x=='simplicity' else x)
        
        sed=df.apply(lambda x: self.get_predictions(x[column_o],x[column_i],x['style_to']), axis=1 )
        c = list(np.array(sed[0]))
        s = list(sed[1])
        u = list(np.array(sed[2]))
        return c,s,u
    


       
        
    
method=LLMProp()
#################################
output_on_dataset_test_both(method)
output_on_dataset_both(method)

