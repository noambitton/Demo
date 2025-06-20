import pandas as pd
import json
import os
import numpy as np
import sys
import re
from system.utils import compute_pareto_front

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

def safe_parse_array(s):
    try:
        # Remove brackets and split by whitespace
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
        return np.array(nums)
    except Exception as e:
        print(f"Failed to parse: {s} — {e}")
        return np.array([])
    
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
            # Columns that contain array-like strings
            bin_cols = [col for col in truth_df.columns if col.endswith("_bins")]
            # Convert string representations to numpy arrays
            for col in bin_cols:
                if col.endswith("_bins"):
                    truth_df[col] = truth_df[col].apply(safe_parse_array)
            
            # Add a new column 'Estimated' based on the Pareto front
            datapoints = [np.array(truth_df['gpt_semantics'].values), np.array(truth_df['utility'].values)]
            lst = compute_pareto_front(datapoints)
            truth_df["Estimated"] = 0
            truth_df.loc[lst, "Estimated"] = 1
            truth_df["Explored"] = 1
            # Rename 'utility' to "Utility"
            truth_df.rename(columns={'utility': 'Utility', 'gpt_semantics':'Semantic'}, inplace=True)
            truth_df.rename(columns={col: col.replace('_bins', '') for col in truth_df.columns if col.endswith('_bins')}, inplace=True)
            # Assign a new ID to each row
            truth_df['ID'] = range(1, len(truth_df) + 1)
            truth_df['ID'] = truth_df['ID'].astype(str)
            truth_df['Estimated'] = truth_df['Estimated'].astype(str)
            truth_df['Explored'] = truth_df['Explored'].astype(str)
        return truth_df
    else:
        raise FileNotFoundError(f"Truth file for dataset {dataset} not found.")