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

def load_truth(dataset, use_case='modeling', attribute=None):
    """
    Load the truth file for the given dataset.
    :param dataset: The name of the dataset.
    :return: DataFrame containing the truth data.
    """
    if use_case == 'visualization':
        if attribute is None:
            raise ValueError("Attribute must be specified for visualization use case.")
        truth_file = os.path.join(PROJECT_DIR, "truth", dataset, f"{dataset}.{attribute}.visualization.csv")
    elif use_case in ['modeling', 'imputation']:
        truth_file = os.path.join(PROJECT_DIR, "truth", dataset, f"{dataset}.multi_attrs.{use_case}.csv")
    else:
        raise ValueError("Invalid use case. Use 'visualization', 'modeling', or 'imputation'.")
    
    if os.path.exists(truth_file):
        if use_case in ['modeling', 'imputation']:
            truth_df = pd.read_csv(truth_file)
            # Rename 'utility' to "Utility"
            truth_df.rename(columns={'utility': 'Utility', 'gpt_semantics':'Semantic'}, inplace=True)
            # Assign a new ID to each row
            truth_df['ID'] = range(1, len(truth_df) + 1)
            truth_df['ID'] = truth_df['ID'].astype(str)
        return truth_df
    else:
        raise FileNotFoundError(f"Truth file for dataset {dataset} not found.")