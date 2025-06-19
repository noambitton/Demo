import pandas as pd
import json
import os
import sys

PROJECT_DIR = os.getcwd()

with open(os.path.join(PROJECT_DIR, "testresults", "Age_spearmanr_gpt_distance.json"), "r") as f:
    vis_binning_options = json.load(f)

with open(os.path.join(PROJECT_DIR, "testresults", "modeling_gpt_distance.json"), "r") as f:
    pred_binning_options = json.load(f)

def naive_seercuts_split(binning_options,task):
    naive_list,seercuts_list=[],[]
    for idx,option in enumerate(binning_options):
        option['ID']=idx
        if task=='pred':
            seercuts_list.append(option)
        elif option['Explored']==1:
            seercuts_list.append(option)
        naive_list.append(option)
    return naive_list,seercuts_list

vis_naive_list,vis_seercuts_list=naive_seercuts_split(vis_binning_options,task='viz')
pred_naive_list,pred_seercuts_list=naive_seercuts_split(pred_binning_options,task='pred')

vis_naive_df=pd.DataFrame(vis_naive_list)
vis_naive_df['Estimated']=vis_naive_df['Estimated'].astype(str)
vis_seercuts_df=pd.DataFrame(vis_seercuts_list)
vis_seercuts_df['Estimated']=vis_seercuts_df['Estimated'].astype(str)
pred_binning_df= pd.DataFrame(pred_seercuts_list)
pred_binning_df['Estimated']=pred_binning_df['Estimated'].astype(str)
# binning_df = pd.DataFrame(binning_options)
