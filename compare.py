import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr,spearmanr,kendalltau
from itertools import permutations
from numpy import random
import itertools
import random 
from data import *
where='data'

                                
def get_valid(data,valid):
    def mean_in(x):
        if x=='None':
            return x
        else:
            return np.mean(x)
    
    # make multi index
    header = ['#size','#annotator','#system','rate_ref','scale','style','pearson-C','pearson-S','pearson-F', 'cor_s_c','cor_s_f','cor_f_c', 'system-rank-C','system-rank-S','system-rank-F']
    pearson_c=mean_in(data.get_annotator_cor('c'))
    pearson_s=mean_in(data.get_annotator_cor('s'))
    pearson_f=mean_in(data.get_annotator_cor('f'))
    size=data.get_size()
    system=data.get_system()
    if 'Reference' in data.df.system.value_counts():
        rate_ref=True 
    else: rate_ref=False
    style = data.get_style()
    scale =list(data.get_scale())
    system_rankC = mean_in(data.get_kendalltau_system('c'))
    system_rankS = mean_in(data.get_kendalltau_system('s'))
    system_rankF = mean_in(data.get_kendalltau_system('f'))
    
    #sample_rankC = mean_in(data.get_kendalltau_sample('c'))
    #sample_rankS = mean_in(data.get_kendalltau_sample('s'))
    #sample_rankF = mean_in(data.get_kendalltau_sample('f'))
    ms_mc=None 
    mf_mc=None
    ms_mf=None
    if 'c-1' in data.df.columns:
        data.get_mean('c')
        if 's-1' in data.df.columns:
            data.get_mean('s')          
            ms_mc =pearsonr(list(data.df['c-mean']),list(data.df['s-mean']))
        if 'f-1' in data.df.columns:
            data.get_mean('f')          
            mf_mc =pearsonr(list(data.df['f-mean']),list(data.df['c-mean']))
            if 's-1' in data.df.columns:
                ms_mf =pearsonr(list(data.df['f-mean']),list(data.df['s-mean']))
            
    df = pd.DataFrame(np.array([[size,data.annotators,system,rate_ref,scale,style,pearson_c,pearson_s,pearson_f,ms_mc,ms_mf,mf_mc,
                                 system_rankC,system_rankS,system_rankF,
                                ]]),
                      index=[data.name], columns=header)
    valid=pd.concat([valid,df])
    return valid

def get_valid_all_reference():
    valid = pd.DataFrame()
    data = Lai3()
    if 'Reference' in data.df.system.value_counts():
        data.keep_reference('Reference')
        valid=get_valid(data,valid)

    data = ScialomD21('reference')
    if 'Reference' in data.df.system.value_counts():
        data.keep_reference('Reference')
        valid=get_valid(data,valid)
    data = ZeigenB11()
    if 'Reference' in data.df.system.value_counts():
        data.keep_reference('Reference')
        valid=get_valid(data,valid)
    return valid
    #print('annotator correlation',np.array(metaeval.get_anno_correlation(system_split=False)).mean())
    #print('correlation metric to annotator',np.array(metaeval.get_correlation(system_split=False)).mean())
    #print('kendalltau metric to annotator',np.array(metaeval.get_kendalltau_system()).mean())
    

    
def output_on_dataset_test_both(method):
    folder = 'output_final'
    column_i='input'
    column_o='output'
    
    def save_output(method,data,folder,column_i,column_o):
        predictsC,predictsS,predictsU=method.predict(data.df,column_i=column_i, column_o=column_o)
        
        dfpred = pd.DataFrame({'pred':predictsC})
        path = 'pred_{}_{}.csv'.format(method.name,data.name)
        savepath = os.path.join(folder,path)
        dfpred.to_csv(savepath)

        dfpredS = pd.DataFrame({'pred':predictsS})
        path = 'pred_style_{}_{}.csv'.format(method.name,data.name)
        savepath = os.path.join(folder,path)
        dfpredS.to_csv(savepath)
        
        if len(predictsU)!=0:
            dfpredU = pd.DataFrame({'pred':predictsU})
            path = 'pred_universal_{}_{}.csv'.format(method.name,data.name)
            savepath = os.path.join(folder,path)
            dfpredU.to_csv(savepath)
    
    # on each data
    data = Syntetisk1()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk2()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk3()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk4()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk5()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk6()
    save_output(method,data,folder,column_i,column_o)
    

def output_on_dataset_both(method):
    folder = 'output_final'
    column_i='input'
    column_o='output'
    
    def save_output(method,data,folder,column_i,column_o):
        predictsC,predictsS,predictsU=method.predict(data.df,column_i=column_i, column_o=column_o)
        
        dfpred = pd.DataFrame({'pred':predictsC})
        path = 'pred_{}_{}.csv'.format(method.name,data.name)
        savepath = os.path.join(folder,path)
        dfpred.to_csv(savepath)

        dfpredS = pd.DataFrame({'pred':predictsS})
        path = 'pred_style_{}_{}.csv'.format(method.name,data.name)
        savepath = os.path.join(folder,path)
        dfpredS.to_csv(savepath)
        if len(predictsU)!=0:
            dfpredU = pd.DataFrame({'pred':predictsU})
            path = 'pred_universal_{}_{}.csv'.format(method.name,data.name)
            savepath = os.path.join(folder,path)
            dfpredU.to_csv(savepath)
    
    # on each data
    data = Mir84()
    save_output(method,data,folder,column_i,column_o)
    
    data = Lai3()
    save_output(method,data,folder,column_i,column_o)
    
    data = Cao6c()
    save_output(method,data,folder,column_i,column_o)
    
    data = ZeigenB11()
    save_output(method,data,folder,column_i,column_o)
    
    data = ScialomD21('system')
    save_output(method,data,folder,column_i,column_o)
    
    data = ScialomD21('reference')
    save_output(method,data,folder,column_i,column_o)
    

def output_on_dataset_test(method):
    folder = 'output_final'
    column_i='input'
    column_o='output'
    
    def save_output(method,data,folder,column_i,column_o):
        predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
        dfpred = pd.DataFrame({'pred':predicts})
        path = 'pred_{}_{}.csv'.format(method.name,data.name)
        savepath = os.path.join(folder,path)
        dfpred.to_csv(savepath)
    
    # on each data
    data = Syntetisk1()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk2()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk3()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk4()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk5()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk6()
    save_output(method,data,folder,column_i,column_o)
    

def output_on_dataset_refmethod(method):
    folder = 'output_final'
    column_i='reference'
    column_o='output'
    method_name=method.name.replace('Self','Ref')
    def save_output(method,data,folder,column_i,column_o):
        predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
        dfpred = pd.DataFrame({'pred':predicts})
        path = 'pred_{}_{}.csv'.format(method_name,data.name)
        savepath = os.path.join(folder,path)
        dfpred.to_csv(savepath)
    
    data = Syntetisk1()
    save_output(method,data,folder,column_i,column_o)
    
    data = Syntetisk2()
    save_output(method,data,folder,column_i,column_o)
       
    data = Lai3()
    data.on_reference('Reference')
    save_output(method,data,folder,column_i,column_o)
    
    
    data = ZeigenB11()
    data.on_reference('Reference')
    save_output(method,data,folder,column_i,column_o)
    
def output_on_dataset(method):
    folder = 'output_final'
    column_i='input'
    column_o='output'
    
    # on each data
    data = Mir84()
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    
    data = Lai3()
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    
    
    data = Cao6c()
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    
    data = ZeigenB11()
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    
    data = ScialomD21('system')
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    
    data = ScialomD21('reference')
    predicts=method.predict(data.df,column_i=column_i, column_o=column_o)
    dfpred = pd.DataFrame({'pred':predicts})
    path = 'pred_{}_{}.csv'.format(method.name,data.name)
    savepath = os.path.join(folder,path)
    dfpred.to_csv(savepath)
    

