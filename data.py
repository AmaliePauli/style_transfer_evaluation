import pandas as pd
import os
import random 
import numpy as np
import itertools
import pandas as pd
import pandas as pd
import numpy as np
import os
from scipy.stats import pearsonr,spearmanr,kendalltau
from itertools import permutations
from numpy import random
import itertools
import random 

where='data'


class RatedData:
    def __init__(self):
        pass

    def get_size(self):
       
        return self.df.shape[0]
    def get_system(self):
        return len(self.df['system'].unique())
    def get_style(self):
        return self.df['style_to'].unique()
    
    def get_mean(self,dim:str):
        
        self.df['{}-mean'.format(dim)]=self.df[['{}-{}'.format(dim,i) for i in np.arange(1,self.annotators+1)]].mean(axis=1)
    def get_median(self,dim:str):
        
        self.df['{}-median'.format(dim)]=self.df[['{}-{}'.format(dim,i) for i in np.arange(1,self.annotators+1)]].median(axis=1)
    
    def get_universal_min(self):
        self.get_mean('s')
        self.get_mean('c')
        if 'f-1' in self.df.columns:
            self.get_mean('f')
            self.df['universal']=self.df[['c-mean','s-mean','f-mean']].min(axis=1)
        self.df['universal']=self.df[['c-mean','s-mean']].min(axis=1)
    def get_universal_mean(self):
        self.get_mean('s')
        self.get_mean('c')
        if 'f-1' in self.df.columns:
            self.get_mean('f')
            self.df['universal']=self.df[['c-mean','s-mean','f-mean']].mean(axis=1)
        self.df['universal']=self.df[['c-mean','s-mean']].mean(axis=1)
        
    def get_scale(self):
        mines=[]
        maxes=[]
        for dim in ['c','s','f']:
            minimum=float('inf')
            maximum=float('-inf')
            for nr in np.array(range(1,self.annotators+1)):
                rater ='{}-{}'.format(dim,nr)
                if rater in self.df.columns:
                    mi=self.df[rater].min()
                    if minimum>mi:
                        minimum=mi
                    ma=self.df[rater].max()
                    if maximum<ma:
                        maximum=ma
            mines.append(minimum)
            maxes.append(maximum)
        return np.array([(mines[i],maxes[i]) for i in range(len(mines))])
    
    def get_annotator_cor(self,dim:str):
        # returning spearson correlation for each system
        if self.annotators==1:
            return 'None'
  
        systems=self.df['system'].unique()
        results= []
        combi=list(itertools.combinations(np.array(range(1,self.annotators+1)), 2))
        for pair in combi:
            rater1 ='{}-{}'.format(dim,pair[0])
            rater2 ='{}-{}'.format(dim,pair[1])
            res=[]
            for sys in systems:       
                res.append(spearmanr(self.df[self.df.system==sys][rater1],self.df[self.df.system==sys][rater2])[0])
            results.append(res)
        return results
    def _get_system_anno_rank(self,dim:str):
        # ranking the different systems 
        dimin = '{}-1'.format(dim)
        if dimin not in self.df.columns:
            return 'None'
        systems=self.df['system'].unique()
        results= []
        for nr in range(1,self.annotators+1):
            raters ='{}-{}'.format(dim,nr)
            res=[]
            for sys in systems:
                res.append(self.df[self.df.system==sys][raters].mean())
            results.append(res)
        return results
    def get_kendalltau_system(self,dim:str):
        if self.annotators==1:
            return 'None'
        anno_rankings = self._get_system_anno_rank(dim)
        results=[]
        combi=list(itertools.combinations(np.array(range(0,self.annotators)), 2))
        for pair in combi:

            results.append(kendalltau(anno_rankings[pair[0]],anno_rankings[pair[1]])[0])

        return results
    
    def get_kendalltau_sample(self,dim:str):
 
        # sample against sample 
        if self.annotators==1:
            return 'None'

        results=[]
        combi=list(itertools.combinations(np.array(range(1,self.annotators+1)), 2))
        for pair in combi:
            rater1 = '{}-{}'.format(dim,pair[0])
            rater2 = '{}-{}'.format(dim,pair[1])
            res=[]
            for sample in self.df.sample_id.unique():
                tau=kendalltau(list(self.df[self.df.sample_id==sample][rater1]),list(self.df[self.df.sample_id==sample][rater2]))[0]
                if not np.isnan(tau):
                    res.append(tau)
            results.append(np.array(res).mean())

        return results
        
        
    def downsize(self,samplesize):
        if samplesize is not None:
            topick=random.sample(list(self.df.sample_id.unique()),10)
            self.df=self.df[[True if (t in topick) else False for t in self.df.sample_id]].reset_index(drop=True)
            
    def on_reference(self,col):
        if col in self.df.system.value_counts():
            self.df = self.df[self.df.system!=col].reset_index(drop=True)
    def keep_reference(self,col):
       
        self.df = self.df[self.df.system==col].reset_index(drop=True)
                
class Syntetisk1(RatedData):
    def __init__(self):
        self.annotators=0
        self.path= os.path.join(where,'syntest_anno1.csv')#'synethic_test.csv')
        self.name='syn'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['c-mean']=df['annoB_mean']
        df['s-mean']=df['annoA_mean']
        return df.reset_index(drop=True)
    
class Syntetisk2(RatedData):
    def __init__(self):
        self.annotators=0
        self.path= os.path.join(where,'syntest_anno2.csv')#'synethic_test.csv')
        self.name='syn2'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['c-mean']=df['annoB_mean']
        df['s-mean']=df['annoA_mean']
        return df.reset_index(drop=True)  
    
class Syntetisk3(RatedData):
    def __init__(self):
        self.annotators=0
        self.path= os.path.join(where,'syntest_anno3.csv')#'synethic_test.csv')
        self.name='syn3'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['c-mean']=df['annoB_mean']
        df['s-mean']=df['annoA_mean']
        return df.reset_index(drop=True)

class Syntetisk4(RatedData):
    def __init__(self):
        self.annotators=0
        self.path= os.path.join(where,'syntest_anno4.csv')#'synethic_test.csv')
        self.name='syn4'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['c-mean']=df['annoB_mean']
        df['s-mean']=df['annoA_mean']
        return df.reset_index(drop=True) 
    
class Syntetisk5(RatedData):
    def __init__(self):
        self.annotators=0
        self.path= os.path.join(where,'syntest_anno5.csv')#'synethic_test.csv')
        self.name='syn5'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['c-mean']=df['annoB_mean']
        df['s-mean']=df['annoA_mean']
        return df.reset_index(drop=True) 
    
class Syntetisk6(RatedData):
    def __init__(self):
        self.annotators=0
        self.path= os.path.join(where,'syntest_anno6.csv')#'synethic_test.csv')
        self.name='syn6'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['c-mean']=df['annoB_mean']
        df['s-mean']=df['annoA_mean']
        return df.reset_index(drop=True)   
    
    
class Mir84(RatedData):
    def __init__(self):
        self.annotators=1
        self.path= os.path.join(where,'mir84','mir84sf.csv')
        self.name='Mir84'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        return df.reset_index(drop=True)
    
    
class Lai3(RatedData):
    def __init__(self):
        self.annotators=2
        #self.path= os.path.join(where,'lai3')
        #self.system = os.listdir(self.path)
        self.path= os.path.join(where,'lai3','combined_data.csv')
        self.df=self.load_data()
        self.name='Lai3'
        
    def load_data(self):
        df = pd.read_csv(self.path)
        return df.reset_index(drop=True)

    
    
class Cao6c(RatedData):
    def __init__(self):
        self.annotators=1
        self.path1= os.path.join(where,'cao6c','expert_to_layman.csv')
        self.path2= os.path.join(where,'cao6c','layman_to_expert.csv')
        self.name='Cao6c'
        self.df=self.load_data()
        
    def load_data(self):
        df1 = pd.read_csv(self.path1)
        df2 = pd.read_csv(self.path2)
        df=pd.concat([df1,df2])
        df['output'] = df['output'].apply(lambda x: str(x))
        df['input'] = df['input'].apply(lambda x: str(x))
        return df.reset_index(drop=True)


    
class ZeigenB11(RatedData):
    def __init__(self):
        self.annotators=5
        self.path=  os.path.join(where,'ZeigenB11','prepare_results.csv')   
        self.name='ZeigenB11'
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path)
        df['style_to']='appropriated'
        return df.reset_index(drop=True)
     
class ScialomD21(RatedData):
    def __init__(self,task):
        if task=='system':
            self.annotators=11
            file = 'prepare_system.csv'
        if task=='reference':
            self.annotators=25
            file = 'prepare_human.csv'
        self.path= os.path.join(where,'scialomD21',file)
        self.name='ScialomD21_{}'.format(task)
        self.df=self.load_data()
        
    def load_data(self):
        df = pd.read_csv(self.path).rename(columns={'sentence_id':'sample_id'})
        df['style_to']='simplified'
        df['system']=df.system.apply(lambda x: 'Reference' if x=='reference' else x)
        return df.reset_index(drop=True)    