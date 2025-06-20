import sys
import os
from joblib import Parallel, delayed
import json
from typing import List, Union, Any, Tuple, Dict
import time
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from scipy.stats import wasserstein_distance, spearmanr, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import squareform

#from discretizers import *
from system.TestSearchSpace import *
#from SearchSpace import *
from system.utils import *

ID_COUNT = 0

def find_optimal_t(search_space) -> float:
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)

    # Convert to condensed form if needed
    condensed_X = squareform(X)  # X must be square and symmetric

    # Compute linkage
    Z = linkage(condensed_X, method='ward')

    # Cophenetic correlation and distances
    coph_corr, coph_dists = cophenet(Z, condensed_X)
    t_values = np.linspace(0.4, 0.8, 20)

    results = []

    for t in t_values:
        labels = fcluster(Z, t=t, criterion='distance')
        n_clusters = len(set(labels))

        if n_clusters <= 1 or n_clusters >= len(X):
            continue

        try:
            sil = silhouette_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            db = davies_bouldin_score(X, labels)
            results.append((t, sil, ch, db))
        except Exception as e:
            print(f"Skipping t={t:.2f} due to error: {e}")

    # Convert to NumPy array for easier processing
    results = np.array(results)
    ts, sils, chs, dbs = results.T

    # Normalize: higher is better for silhouette and CH; lower is better for DB
    sils_norm = minmax_scale(sils)
    chs_norm = minmax_scale(chs)
    dbs_norm = minmax_scale(-dbs)  # invert so that lower DB → higher score

    # Weighted average of normalized metrics
    composite_score = 1 * sils_norm + 0 * chs_norm + 0 * dbs_norm
    best_idx = np.argmax(composite_score)

    # Display the best t
    print(f"t = {ts[best_idx]:.2f}")
    return ts[best_idx]

def explainable_modeling_using_strategy(data, y_col, strategy) -> float:
    """
    Wrapper function to model the data using an explainable model
    ***** Note: This function is only used in demo_data_modeling_case.ipynb for now *****
    :param data: DataFrame
    :param y_col: str
    :param partition_dict: Dict[str, Partition]
    :return: float
    """
    attributes = []
    for partition in strategy.partition_list:
        attr = partition.attribute
        attributes.append(attr)
        #print(f"Discretizing {attr}...")
        data[attr + '.binned'] = pd.cut(data[attr], bins=partition.bins, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])
    
    #print(f"Data shape after binning: {data.shape}")
    
    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col not in attributes]
    X = data[X_cols]
    y = data[y_col]
    #print(f"Data shape before train: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    return model_accuracy

def visualization_one_attr(data, y_col, attr:str, partition) -> float:
    """
    Wrapper function to visualize the data using ANOVA
    """
    start_time = time.time()
    data = data[[attr, y_col]]
    data[attr] = pd.cut(data[attr], bins=partition.bins, labels=partition.bins[1:])
    data[attr] = data[attr].astype('float64')
    data = data.groupby(attr)[y_col]
    data = [group[1] for group in data]
    f, p = f_oneway(*data)
    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = f
    return f

def explainable_modeling_one_attr(data, y_col, attr:str, partition) -> float:
    """
    Wrapper function to model the data using an explainable model
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
    data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
    data = data.dropna(subset=[attr + '.binned'])
    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col != attr]
    X = data[X_cols]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = model_accuracy
    return model_accuracy

def data_imputation_one_attr(data, y_col, attr:str, partition, imputer='KNN') -> float:
    """
    Wrapper function to impute missing values in a dataset
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    bins = partition.bins
    # Bin attr column, with nan values
    data[attr + '.binned'] = pd.cut(data[attr], bins=bins, labels=bins[1:])
    
    # Impute the missing values using KNN
    X_cols = [col for col in data.columns if col != y_col and col != attr and col != attr + '.gt']
    X = data[X_cols]
    idx = X.columns.get_loc(attr + '.binned')
    imputer = KNNImputer(n_neighbors=len(bins)-1)
    X_imputed = imputer.fit_transform(X)
    
    # Bin imputed values
    data_imputed = np.round(X_imputed[:, idx])
    data[attr+'.imputed'] = data_imputed
    data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=bins, labels=bins[1:])
    data[attr + '.final'] = data[attr + '.final'].astype('float64')
    value_final = np.array(data[attr + '.final'].values)
    value_final[np.isnan(value_final)] = -1
    value_final = np.round(value_final)

    # Evaluate data imputation
    data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=bins, labels=bins[1:])
    data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
    value_gt = np.array(data[attr + '.gt'].values)
    value_gt[np.isnan(value_gt)] = -1
    value_gt = np.round(value_gt)
    impute_accuracy = accuracy_score(value_gt, value_final)

    partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    partition.utility = impute_accuracy
    return impute_accuracy

def explainable_modeling_multi_attrs(data, y_col, partition_dict) -> float:
    """
    Wrapper function to model the data using an explainable model
    ***** Note: This function is only used in demo_data_modeling_case.ipynb for now *****
    :param data: DataFrame
    :param y_col: str
    :param partition_dict: Dict[str, Partition]
    :return: float
    """
    start_time = time.time()
    
    for attr, partition in partition_dict.items():
        #print(f"Discretizing {attr}...")
        data[attr + '.binned'] = pd.cut(data[attr], bins=partition, labels=False)
        data[attr + '.binned'] = data[attr + '.binned'].astype('float64')
        data = data.dropna(subset=[attr + '.binned'])
    
    
    # Evaluate the explainable model
    X_cols = [col for col in data.columns if col != y_col and col not in list(partition_dict.keys())]
    X = data[X_cols]
    y = data[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a decision tree model
    model = DecisionTreeClassifier(random_state=0, max_depth=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    return model_accuracy

def data_imputation_multi_attrs(data, y_col, partition_dict, imputer='KNN') -> float:
    """
    Wrapper function to impute missing values in a dataset
    :param data: DataFrame
    :param attr: str
    :param bins: List
    :return: float
    """
    start_time = time.time()
    # Bin attr column, with nan values
    for attr, partition in partition_dict.items():
        #print(f"Discretizing {attr}...")
        data[attr + '.binned'] = pd.cut(data[attr], bins=partition, labels=False)
        
    # Impute the missing values using KNN
    gt_columns = [attr + '.gt' for attr in partition_dict.keys()]
    X_cols = [col for col in data.columns if col != y_col and col not in list(partition_dict.keys()) and col not in gt_columns]
    X = data[X_cols]
    avg_length = np.mean([len(partition) for partition in partition_dict.values()])
    if imputer == 'KNN':
        imputer = KNNImputer(n_neighbors= int(avg_length))
        X_imputed = imputer.fit_transform(X)
    elif imputer == 'Iterative':
        imputer = IterativeImputer(max_iter=3, random_state=0)
        X_imputed = imputer.fit_transform(X)
    elif imputer == 'Simple':
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
    else:
        raise ValueError(f"Unknown imputer: {imputer}")
    
    # Bin imputed values
    accuracy = []
    #data_imputed = np.round(X_imputed[:, idx])
    for attr, partition in partition_dict.items():
        idx = X.columns.get_loc(attr + '.binned')
        data[attr+'.imputed'] = X_imputed[:, idx]
        data[attr + '.final'] = pd.cut(data[attr+'.imputed'], bins=partition, labels=partition[1:])
        data[attr + '.final'] = data[attr + '.final'].astype('float64')
        value_final = np.array(data[attr + '.final'].values)
        value_final[np.isnan(value_final)] = -1
        value_final = np.round(value_final)

        # Evaluate data imputation
        data[attr + '.gt'] = pd.cut(data[attr + '.gt'], bins=partition, labels=partition[1:])
        data[attr + '.gt'] = data[attr + '.gt'].astype('float64')
        value_gt = np.array(data[attr + '.gt'].values)
        value_gt[np.isnan(value_gt)] = -1
        value_gt = np.round(value_gt)
        impute_accuracy = accuracy_score(value_gt, value_final)
        accuracy.append(impute_accuracy)

    #partition.f_time.append((partition.ID, 'utility_comp', time.time() - start_time))
    #partition.utility = impute_accuracy
    return np.mean(accuracy)

def get_runtime_stats(search_space, semantic_metric='l2_norm', indices=None) -> List:
    # Get runtime statistics
    runtime_stats = []
    runtime_df = search_space.get_runtime()
    partition_gen = runtime_df[runtime_df['function'] == 'get_bins']['runtime'].sum()
    if indices is not None:
        runtime_df = runtime_df[runtime_df['ID'].isin(indices)]
        num_explored_points = len(indices)
    else:
        num_explored_points = len(search_space.candidates)
    
    runtime_stats.append(num_explored_points)
    runtime_stats.append(partition_gen)
    functions = [f'cal_{semantic_metric}', 'utility_comp']
    for f in functions:
        total_time = runtime_df[runtime_df['function'] == f]['runtime'].sum()
        runtime_stats.append(total_time)
    return runtime_stats

def DBSCAN_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :return: List of clusters
    """
    eps = parameters['eps']
    min_samples = parameters['min_samples']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    dbscan_clusters = model.fit_predict(X)
    # For outliers, assign them to each of their separate cluster
    max_cluster = np.max(dbscan_clusters)
    for i, c in enumerate(dbscan_clusters):
        if c == -1:
            dbscan_clusters[i] = max_cluster + 1
            max_cluster += 1
    return dbscan_clusters

#def HDBSCAN_distributions(search_space, parameters) -> List:
#    """
#    :param search_space: PartitionSearchSpace
#    :return: List of clusters
#    """
#    min_cluster_size = parameters['min_cluster_size']
#    X = np.array([p.distribution for p in search_space.candidates])
#    X = pairwise_distance(X, metric=wasserstein_distance)
#    model = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=min_cluster_size)
#    hdbscan_clusters = model.fit_predict(X)
#    # For outliers, assign them to each of their separate cluster
#    max_cluster = np.max(hdbscan_clusters)
#    for i, c in enumerate(hdbscan_clusters):
#        if c == -1:
#            hdbscan_clusters[i] = max_cluster + 1
#            max_cluster += 1
#    return hdbscan_clusters

#def HDBSCAN_binned_values(search_space, parameters) -> List:
#    """
#    :param search_space: PartitionSearchSpace
#    :return: List of clusters
#    """
#    n_components = parameters['n_components']
#    X = np.array([p.binned_values for p in search_space.candidates])
#    model = hdbscan.HDBSCAN(min_cluster_size=2)
#    pca = PCA(n_components=n_components)
#    X = pca.fit_transform(X)
#    hdbscan_clusters = model.fit_predict(X)
#    return hdbscan_clusters

def linkage_distributions(search_space, parameters) -> List:
    """
    :param search_space: PartitionSearchSpace
    :param parameters: Dict
    :return: List of clusters
    """
    t = parameters['t']
    criterion = parameters['criterion']
    X = np.array([p.distribution for p in search_space.candidates])
    X = pairwise_distance(X, metric=wasserstein_distance)
    Z = linkage(X, method='ward')
    agg_clusters = fcluster(Z, t=t, criterion=criterion)
    #agg_clusters = fcluster(Z, t=0.5, criterion='distance')
    agg_clusters = [x-1 for x in agg_clusters] # 0-indexing
    return agg_clusters

def parallel_pdist(X):
    """Parallel version of pdist using joblib"""
    n = len(X)
    dists = Parallel(n_jobs=-1)(
        delayed(wasserstein_distance_strategy)(X[i], X[j]) 
        for i in range(n) for j in range(i+1, n)
    )
    return dists

def linkage_strategies(strategy_space, parameters) -> list:
    t = parameters['t']
    criterion = parameters['criterion']

    X = [tuple(p.distribution for p in s.partition_list) for s in strategy_space.candidates]

    # Compute pairwise Wasserstein distances
    #X = pdist(X, metric=wasserstein_distance_strategy)

    # Parallelized distance computation
    X = parallel_pdist(X)

    Z = linkage(X, method='ward')
    agg_clusters = fcluster(Z, t=t, criterion=criterion)

    return [x - 1 for x in agg_clusters]

def wasserstein_distance_strategy(s1, s2) -> float:
    """
    Compute the average Wasserstein distance between corresponding distributions in s1 and s2.
    
    :param s1: List of distributions (arrays) for strategy 1
    :param s2: List of distributions (arrays) for strategy 2
    :return: Average Wasserstein distance
    """
    return np.mean([wasserstein_distance(a, b) for a, b in zip(s1, s2)])

def random_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    sampled_indices = []
    print("Budget start:", budget)
    print("Unique clusters:", np.unique(cluster_assignments))
    if budget >= len(np.unique(cluster_assignments)):
        # Only sample one partition from each cluster
        for c in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == c)[0]
            # Sample one partition from the cluster
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
        if budget > len(np.unique(cluster_assignments)):
            assignments = cluster_assignments.copy()
            # Remove sampled_indices from the cluster assignments by index
            new_list_to_sample = [item for i, item in enumerate(assignments) if i not in sampled_indices]
            sampled_indices.extend(np.random.choice(new_list_to_sample, budget - len(sampled_indices), replace=False))
        # Add gold standard partition to the sampled partitions
        if 0 not in sampled_indices:
            sampled_indices.append(0)
    
    return sampled_indices

def random_sampling_clusters_robust(cluster_assignments, parameters) -> List:
    """
    Sample partitions from clusters with a budget constraint.
    Similar to random_sampling_clusters, but more robust.
    In the sense that if the budget (n) is less than the number of clusters,
    this method will sample n clusters and sample one partition from each cluster.
    """
    p = parameters['p']
    budget = max(int(len(cluster_assignments) * p) - 1, 1)
    sampled_indices = []
    if budget >= len(np.unique(cluster_assignments)):
        sampled_indices = random_sampling_clusters(cluster_assignments, parameters)
    
    # Sample clusters when budget is less than the number of clusters
    else:
        sampled_clusters = np.random.choice(np.unique(cluster_assignments), budget, replace=False)
        for c in sampled_clusters:
            cluster_indices = np.where(cluster_assignments == c)[0]
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def random_with_inverse_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    sampled_indices = []
    print("Budget start:", budget)
    print("Unique clusters:", np.unique(cluster_assignments))
    if budget >= len(np.unique(cluster_assignments)):
        # Only sample one partition from each cluster
        for c in np.unique(cluster_assignments):
            cluster_indices = np.where(cluster_assignments == c)[0]
            # Sample one partition from the cluster
            sampled_indices.extend(np.random.choice(cluster_indices, 1, replace=False))
        
        if budget > len(np.unique(cluster_assignments)):
            assignments = cluster_assignments.copy()
            # Remove sampled_indices from the cluster assignments by index
            new_cluster_assignments = [item for i, item in enumerate(assignments) if i not in sampled_indices]
            #sampled_indices.extend(np.random.choice(new_list_to_sample, budget - len(sampled_indices), replace=False))
            budget = budget - len(sampled_indices)
            # Calculate cluster size from cluster assignment
            cluster_size = [len(np.where(new_cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
            cluster_probs = 1 / (cluster_size / np.sum(cluster_size))
            cluster_probs = cluster_probs / np.sum(cluster_probs)
            cluster_probs = np.nan_to_num(cluster_probs)
            # get number of samples per cluster, with at least one sample per cluster
            cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
            # sample from each cluster based on the number of samples
            for c in np.unique(new_cluster_assignments):
                cluster_indices = np.where(new_cluster_assignments == c)[0]
                sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))

        
        # Add gold standard partition to the sampled partitions
        if 0 not in sampled_indices:
            sampled_indices.append(0)
    
    return sampled_indices
    
def proportional_sampling_clusters(cluster_assignments, parameters) -> List:
    # Proportionally sample from each cluster
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    clusters = [i for i in range(len(np.unique(cluster_assignments)))]
    # get cluster probabilities
    cluster_probs = np.bincount(cluster_assignments) / len(cluster_assignments)
    cluster_size = [len(np.where(cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
    # get number of samples per cluster, with at least one sample per cluster
    cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
    # get number of samples per cluster, with at least one sample per cluster
    #cluster_samples = [0] * len(np.unique(cluster_assignments))
    #samples = np.random.choice(clusters, p=cluster_probs, size=budget)
    #for c in samples: cluster_samples[c] += 1
    # sample from each cluster based on the number of samples
    sampled_indices = []
    for c in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == c)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))

    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def find_actual_cluster_sample_size(total_budget, norm_inv_probs, cluster_sizes):

    # Step 3: Calculate the ideal number of samples for each cluster
    # Based on the inverse probabilities and total budget
    ideal_samples = [int(p * total_budget) for p in norm_inv_probs]
    ideal_excess = sum(ideal_samples) - total_budget
    #print("Inv probs:", norm_inv_probs)
    #print("Ideal samples:", ideal_samples)

    # Step 4: Initialize an array to track the actual samples drawn from each cluster
    actual_samples = [0] * len(norm_inv_probs)

    # Step 5: First pass: Assign as many samples as possible without exceeding cluster capacity
    excess_budget = 0  # Track how much of the budget is left after clusters with limited points
    for i in range(len(norm_inv_probs)):
        if ideal_samples[i] <= cluster_sizes[i]:
            # We can sample the ideal number from this cluster
            actual_samples[i] = ideal_samples[i]
        else:
            # Not enough points in this cluster, so sample all available points
            actual_samples[i] = cluster_sizes[i]
            # Add the remaining unused budget
            excess_budget += ideal_samples[i] - cluster_sizes[i]
    if ideal_excess < 0: excess_budget -= ideal_excess
    #print("Excess budget:", excess_budget)
    
    # Step 6: Redistribute the excess budget
    # Only distribute to clusters that still have points left to sample
    prev_excess_budget = 0
    remaining_inv_probs, remaining_inv_sum = [], 0
    clusters = [i for i in range(len(cluster_sizes))]
    while excess_budget > 0 and excess_budget != prev_excess_budget:
        #print("Excess budget:", excess_budget)
        remaining_inv_probs = [inv_p if actual_samples[i] < cluster_sizes[i] else 0 for i, inv_p in enumerate(norm_inv_probs)]
        remaining_inv_sum = sum(remaining_inv_probs)
        #print("Remaining inv probs:", np.array(remaining_inv_probs) / remaining_inv_sum)
        # remove clusters that have been fully sampled
        clusters = [i for i in range(len(cluster_sizes)) if remaining_inv_probs[i] > 0]
        remaining_inv_probs = [inv_p for i, inv_p in enumerate(remaining_inv_probs) if inv_p > 0]
        
        if remaining_inv_sum == 0:
            break  # No more clusters to redistribute to
        
        additionals = [0] * len(norm_inv_probs)
        samples = np.random.choice(clusters, p=np.array(remaining_inv_probs)/remaining_inv_sum, size=excess_budget)
        for c in samples: additionals[c] += 1

        for i in range(len(norm_inv_probs)):
            if actual_samples[i] < cluster_sizes[i]:
                # Compute additional samples to allocate
                additional_samples = additionals[i]
                #print("Additional samples:", additional_samples)
                # Ensure we don't exceed the cluster's capacity
                available_capacity = cluster_sizes[i] - actual_samples[i]
                
                if additional_samples <= available_capacity:
                    actual_samples[i] += additional_samples
                    prev_excess_budget = excess_budget
                    excess_budget -= additional_samples
                else:
                    # Take all remaining points from the cluster and update the excess budget
                    actual_samples[i] += available_capacity
                    prev_excess_budget = excess_budget
                    excess_budget -= available_capacity
    
    # Output: The final number of samples to draw from each cluster
    #print("Actual samples:", actual_samples)
    return actual_samples


def reverse_propotional_sampling_clusters(cluster_assignments, parameters) -> List:
    p = parameters['p']
    budget = int(len(cluster_assignments) * p)
    #print("Budget start:", budget)
    cluster_probs = 1 / (np.bincount(cluster_assignments) / len(cluster_assignments))
    cluster_probs = cluster_probs / np.sum(cluster_probs)
    # Calculate cluster size from cluster assignment
    cluster_size = [len(np.where(cluster_assignments == c)[0]) for c in np.unique(cluster_assignments)]
    sampled_indices = []
    # get number of samples per cluster, with at least one sample per cluster
    cluster_samples = find_actual_cluster_sample_size(budget, cluster_probs, cluster_size)
    # sample from each cluster based on the number of samples
    for c in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == c)[0]
        sampled_indices.extend(np.random.choice(cluster_indices, cluster_samples[c], replace=False))
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def reverse_propotional_sampling_clusters_(cluster_assignments, parameters) -> List:
    budget = int(len(cluster_assignments) * 0.2)
    # order the clusters by size
    cluster_sizes = np.bincount(cluster_assignments)
    sorted_clusters = np.argsort(cluster_sizes)
    sampled_indices = []
    for c in sorted_clusters:
        c_budget = budget - len(sampled_indices)
        cluster_indices = np.where(cluster_assignments == c)[0]
        if len(cluster_indices) > c_budget:
            sampled_indices.extend(np.random.choice(cluster_indices, c_budget, replace=False))
        else: sampled_indices.extend(cluster_indices)
        if len(sampled_indices) >= budget:
            break
    
    # Add gold standard partition to the sampled partitions
    if 0 not in sampled_indices:
        sampled_indices.append(0)
    return sampled_indices

def cluster_sampling(search_space, clustering, sampling, semantic_metric='l2_norm', clustering_params:Dict={}, sampling_params:Dict={}, if_runtime_stats=True) -> List:
    runtime_stats = []

    # Cluster the binned data
    start_time = time.time()
    
    cluster_assignments = clustering(search_space, clustering_params)

    sampled_indices = sampling(cluster_assignments, sampling_params)
    if len(sampled_indices) == 0:
        return None, None, runtime_stats, cluster_assignments
    
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)
    
    method_comp = time.time() - start_time
    #points_df['Cluster'] = cluster_assignments

    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats, cluster_assignments

def random_sampling(search_space, semantic_metric='l2_norm', frac=0.5, if_runtime_stats=True) -> List:
    runtime_stats = []
    start_time = time.time()
    # Sample frac of the partitions
    sampled_indices = np.random.choice(len(search_space.candidates), int(len(search_space.candidates) * frac), replace=False)
    sampled_partitions = [search_space.candidates[i] for i in sampled_indices]
    explored_points, pareto_points, _ = get_pareto_front(sampled_partitions, semantic_metric)

    method_comp = time.time() - start_time
    # Compute the runtime statistics
    if if_runtime_stats:
        runtime_stats.extend(get_runtime_stats(search_space, semantic_metric, sampled_indices))
        runtime_stats.append(method_comp)
    return explored_points, pareto_points, runtime_stats

def get_points(partitions, semantic_metric) -> List:
    if semantic_metric == 'l2_norm':
        semantics = [p.l2_norm for p in partitions]
    elif semantic_metric == 'gpt_semantics':
        semantics = [p.gpt_semantics for p in partitions]
    elif semantic_metric == 'KLDiv':
        semantics = [p.KLDiv for p in partitions]
    else: raise ValueError("Invalid semantic metric")
    utility = [p.utility for p in partitions]
    datapoints = [np.array(semantics), np.array(utility)]
    return datapoints

def get_pareto_front(candidates, semantic_metric='l2_norm') -> List:
    datapoints = get_points(candidates, semantic_metric)
    IDs = [p.ID for p in candidates]
    #print(f"Data points: {datapoints}")
    #print("Datapoint shape to compute Pareto points:", np.array(datapoints).shape)
    lst = compute_pareto_front(datapoints)

    # Plot the Pareto front
    pareto_df = pd.DataFrame({'ID': IDs, 'Semantic': datapoints[0], 'Utility': datapoints[1]})
    pareto_df['Pareto'] = 0
    pareto_df.loc[lst, 'Pareto'] = 1
    # TODO: Add the partition to the dataframe, robust to StrategySpace
    if hasattr(candidates[0], "bins"):
        pareto_df['Partition'] = [[p.bins] for p in candidates]
    else: # Strategy: candidates is List[Strategy]
        partition_col_bins = []
        for strategy in candidates:
            bins = [p.bins for p in strategy.partition_list]
            partition_col_bins.append(bins)
        pareto_df['Partition'] = partition_col_bins
    pareto_points = pareto_df[pareto_df['Pareto'] == 1][['Semantic', 'Utility']]
    pareto_points = pareto_points.values.tolist()
    #print(f"Pareto points: {pareto_points}")
    return datapoints, pareto_points, pareto_df
