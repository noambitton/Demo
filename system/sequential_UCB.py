"""
This script is to run sequential UCB on multiple attributes.
"""
import sys
import os
import itertools
from collections.abc import Mapping, Sequence
from datetime import datetime
sys.path.append("./")

from system.utils import *
from system.UCB import *
from system.framework_utils import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

SEMANTICS = ['l2_norm', 'KLDiv', 'gpt_semantics']

def freeze(obj):
    """Recursively turn lists → tuples and dicts → frozensets."""
    if isinstance(obj, Mapping):
        return frozenset((k, freeze(v)) for k, v in obj.items())
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return tuple(freeze(v) for v in obj)
    else:
        return obj  # already hashable

def dedup_dicts(dict_list):
    seen      = set()
    dedup_out = []
    for d in dict_list:
        key = freeze(d)
        if key not in seen:
            seen.add(key)
            dedup_out.append(d)
    return dedup_out


def explainable_modeling_multi_attrs(data, y_col, attrs_bins:Dict) -> float:
    """
    Wrapper function to model the data using an explainable model
    :param data: DataFrame
    :param attrs_bins: Dict
    :return accuracy: float
    """
    for attr, bins in attrs_bins.items():
        data[attr + '.binned'] = pd.cut(data[attr], bins, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])
    
    #print(f"Data shape after binning: {data.shape}")

    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col not in list(attrs_bins.keys())]
    #print(f"X_cols: {X_cols}")
    X = data[X_cols]
    y = data[y_col]
    #print(f"Data shape before train: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Accuracy: {accuracy}")
    return accuracy


def get_semantic_score(candidate, metric:SyntaxWarning):
    """
    Get the semantic score of a candidate
    :param candidate: Candidate object
    :param metric: Semantic metric
    :return: Semantic score
    """
    if metric == 'l2_norm':
        return candidate.l2_norm
    elif metric == 'KLDiv':
        return candidate.KLDiv
    elif metric == 'gpt_semantics':
        return candidate.gpt_semantics
    else:
        raise ValueError(f"Unknown semantic metric: {metric}")

def run_sequential_ucb(project_root, exp_config, use_case='modeling'):
    imputer = 'KNN'  # 'KNN', 'Simple', 'Iterative'
    missing_data_fraction = 0.3 # fraction of data to be used for imputation
    semantic_metric = 'gpt_semantics' #'KLDiv', 'gpt_semantics'
    alpha = 2 # Exploration parameter for UCB
    p = 0.3
    t = 0.4 # Automatic tune when set to 0.0, otherwise set to a fixed value
    
    # read json file
    dataset = exp_config['dataset']
    attributes = exp_config['attributes'].keys()
    gold_standard=exp_config['attributes']
    y_col = exp_config['target']
    raw_data = load_raw_data(project_root, exp_config, use_case=use_case, missing_data_fraction=missing_data_fraction)

    # Make a new folder to save the results
    result_dst_dir = os.path.join(project_root, 'testresults', f"seq_UCB.{use_case}.{semantic_metric}")
    os.makedirs(result_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dst_dir, "cached_results"), exist_ok=True)

    search_space = {}
    data_dfs = {}
    for attr in attributes:
        data = pd.read_csv(os.path.join(project_root, 'data', dataset, 'scored_attributes', f'{attr}.csv'))
        data_dfs[attr] = data
        search_space[attr] = TestSearchSpace(data, attr, decimal_place=8)

    permutations = list(itertools.permutations(attributes))
    
    t_dict = {}
    
    numbers_samples = []
    n_strategies_tried = 0
    est_partitions = []
    for permutation in permutations:
    
        est_partition_dicts = []
        processed_attributes = []
        for n, attr in enumerate(permutation):
            data_i = raw_data.copy()
            #print("====== Attribute:", attr, "======")
            if len(processed_attributes) == 0:
                cached_ID = random.randrange(100000, 1000000)
                method = UCB(
                    gt_data=None,
                    data=data_i,
                    alpha=alpha,
                    p=p,
                    y_col=y_col,
                    search_space=search_space[attr],
                    gold_standard=gold_standard,
                    semantic_metric=semantic_metric,
                    use_case=use_case,
                    t=t,
                    result_dst_dir=result_dst_dir,
                    debug=False
                )
                results, cached_results = method.run(cached_ID=cached_ID, n_runs=1)
                n_strategies_tried += len(search_space[attr].candidates) * p
                t_dict[attr] = method.t
                # Estimated Pareto points only
                est_pareto_partitions = cached_results[0]['estimated_partitions']
                for est_partition in est_pareto_partitions:
                    est_partition_dicts.append({attr: est_partition})
                processed_attributes.append(attr)
            else:
                new_partition_dicts = []
                for partition_dict in est_partition_dicts:
                    data_i = raw_data.copy()
                    for attr2, partition in partition_dict.items():
                        data_i[attr2 + '.binned'] = pd.cut(data_i[attr2], bins=partition[attr2], labels=False)
                        data_i[attr2 + '.binned'] = data_i[attr2 + '.binned'].astype('float64')
                        data_i = data_i.dropna(subset=[attr2 + '.binned'])
                    cached_ID = random.randrange(100000, 1000000)
                    method = UCB(
                        gt_data=None,
                        data=data_i,
                        alpha=alpha,
                        p=p,
                        y_col=y_col,
                        search_space=search_space[attr],
                        gold_standard=gold_standard,
                        semantic_metric=semantic_metric,
                        use_case=use_case,
                        t=t,
                        result_dst_dir=result_dst_dir,
                        debug=False
                    )
                    results, cached_results = method.run(cached_ID=cached_ID, n_runs=1)
                    n_strategies_tried += len(search_space[attr].candidates) * p
                    t_dict[attr] = method.t
                    # Estimated Pareto points only
                    est_pareto_partitions = cached_results[0]['estimated_partitions']
                    for est_partition in est_pareto_partitions:
                        new_partition_dict = partition_dict.copy()
                        new_partition_dict[attr] = est_partition
                        new_partition_dicts.append(new_partition_dict)
                processed_attributes.append(attr)
                est_partition_dicts = new_partition_dicts
        # extend the est_partitions with the new partition dicts
        est_partitions.extend(est_partition_dicts)
        
    # Exhausitve search for the pareto curve partitions
    utility_scores = []
    semantic_scores = []
    new_partition_dicts = []
    for partition_dict in est_partitions:
        new_partition_dict = {}
        semantic_score = 0
        for attr, partition in partition_dict.items():
            new_partition_dict[attr] = partition[attr]
            semantic_score += partition['__semantic__']
        semantic_score /= len(new_partition_dict)
        data_i = raw_data.copy()
        if use_case == 'modeling':
            utility_score = explainable_modeling_multi_attrs(data_i, y_col, new_partition_dict)
        else: 
            utility_score = data_imputation_multi_attrs(data_i, y_col, new_partition_dict, imputer=imputer)
        #print(f"Partition: {new_partition_dict}, Semantic: {semantic_score}, Utility: {utility_score}")
        new_partition_dict['Semantic'] = semantic_score
        new_partition_dict['Utility'] = utility_score
        #new_partition_dict['ID'] = len(new_partition_dicts) + 1  # Assign a unique ID
        new_partition_dict['Explored'] = 1  # Mark as explored
        new_partition_dict['Estimated'] = 0  # Initially not estimated
        #new_partition_dicts.append({'Partition': new_partition_dict, 'Semantic': semantic_score, 'Utility': utility_score, 'Explored': 1, 'Estimated': 0})
        # Sort new_partition_dict keys to ensure consistent order
        new_partition_dict = {k: new_partition_dict[k] for k in sorted(new_partition_dict.keys())}
        new_partition_dicts.append(new_partition_dict)
        utility_scores.append(utility_score)
        semantic_scores.append(semantic_score)
    
    datapoints = np.array([np.array(semantic_scores), np.array(utility_scores)])
    lst = compute_pareto_front(datapoints)
    #est_partitions = [new_partition_dicts[i] for i in lst]
    est_partitions = []
    # This is for demo purposes, we will set the Estimated flag to 1 for the estimated partitions
    for i in lst:
        new_partition_dicts[i]['Estimated'] = 1
        est_partitions.append(new_partition_dicts[i])
    estimated_points = np.array([np.array([p["Semantic"] for p in est_partitions]),
                        np.array([p["Utility"] for p in est_partitions])])
    estimated_points = np.array(estimated_points).T
    estimated_points = np.unique(estimated_points, axis=0)
    print("Final Estimated of This Run:", estimated_points)

    # Turn new_partition_dicts into a dataframe
    new_partition_dicts = dedup_dicts(new_partition_dicts)
    for i, partition_dict in enumerate(new_partition_dicts):
        partition_dict['ID'] = i + 1
    partition_df = pd.DataFrame(new_partition_dicts)
    print(partition_df.head())
    # Drop duplicates based on rows
    partition_df['ID'] = partition_df['ID'].astype(str)
    partition_df['Estimated'] = partition_df['Estimated'].astype(str)
    return partition_df, n_strategies_tried

if __name__ == '__main__':
    ppath = sys.path[0] + '/../'
    dataset = 'pima'
    use_case = 'modeling'  # 'modeling'
    exp_config = json.load(open(os.path.join(ppath, 'data', dataset, f'{dataset}.json')))
    df = run_sequential_ucb(ppath, exp_config, use_case=use_case)
    print(f"Number of estimated partitions: {len(df)}")
    df.head()