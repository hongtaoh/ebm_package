from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
import os
from scipy.stats import kendalltau
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import math

def compute_theta_phi_for_biomarker(
    biomarker_df: pd.DataFrame,
) -> Tuple[float, float, float, float]:
    """
    Calculate the mean and standard deviation of theta and phi parameters 
    for a specified biomarker using clustering techniques.

    Use KMeans first as default and then reassign clusters using Agglomerative 
    Clustering if healthy participants are not in a single cluster after KMeans

    Args:
    biomarker_df (pd.DataFrame): DataFrame containing participant data for a specific biomarker 
        with columns 'participant', 'biomarker', 'measurement', and 'diseased'.

    Returns:
    tuple: A tuple containing the mean and standard deviation of theta and phi:
        - theta_mean (float): Mean of the measurements in the theta cluster.
        - theta_std (float): Standard deviation of the measurements in the theta cluster.
        - phi_mean (float): Mean of the measurements in the phi cluster.
        - phi_std (float): Standard deviation of the measurements in the phi cluster.
    """
    # Ensure n_init is set properly (use "auto" or an integer)
    n_init_value = int(10)
    clustering_setup = KMeans(n_clusters=2, n_init=n_init_value)

    # you need to make sure each measurment is a np.array before putting it into "fit"
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)

    # Fit clustering method
    clustering_result = clustering_setup.fit(measurements)
    predictions = clustering_result.labels_

    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    diseased_df = biomarker_df[biomarker_df['diseased'] == True]
    # healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)
    # which cluster are healthy participants in
    healthy_predictions = predictions[healthy_df.index]
    diseased_predictions = predictions[diseased_df.index]

    # the mode of the above predictions will be the phi cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx

    if len(set(healthy_predictions)) <= int(1) or len(set(diseased_predictions)) <= int(1):
        clustering = AgglomerativeClustering(n_clusters=2).fit(
            measurements)
        updated_predictions = clustering.labels_
    else:
        updated_predictions = predictions.copy()

    # if len(set(healthy_predictions)) > 1:
    #     # Reassign clusters using Agglomerative Clustering
    #     clustering = AgglomerativeClustering(
    #         n_clusters=2).fit(healthy_measurements)

    #     # Find the dominant cluster for healthy participants
    #     phi_cluster_idx = mode(clustering.labels_, keepdims=False).mode

    #     # Update predictions to ensure all healthy participants are in the dominant cluster
    #     updated_predictions = predictions.copy()
    #     for i in healthy_df.index:
    #         updated_predictions[i] = phi_cluster_idx
    # else:
    #     updated_predictions = predictions


    # two empty clusters to strore measurements
    clusters = [[] for _ in range(2)]
    # Store measurements into their cluster
    for i, prediction in enumerate(updated_predictions):
        clusters[prediction].append(measurements[i][0])

    # Calculate means and standard deviations
    theta_mean, theta_std = np.mean(
        clusters[theta_cluster_idx]), np.std(clusters[theta_cluster_idx])
    phi_mean, phi_std = np.mean(clusters[phi_cluster_idx]), np.std(
        clusters[phi_cluster_idx])

    # check whether the prior_theta_phi contain 0s or nan
    if theta_std == 0 or math.isnan(theta_std):
        print(f"In prior_theta_phi, theta_std is {theta_std}")
    if phi_std == 0 or math.isnan(phi_std):
        print(f"In prior_theta_phi, phi_std is {phi_std}")

    # check whether the prior_theta_phi contain 0s or nan
    if theta_mean == 0 or math.isnan(theta_mean):
        print(f"In prior_theta_phi, theta_mean is {theta_mean}")
    if phi_mean == 0 or math.isnan(phi_mean):
        print(f"In prior_theta_phi, phi_mean is {phi_mean}")

    return theta_mean, theta_std, phi_mean, phi_std


def get_theta_phi_estimates(
    data: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Obtain theta and phi estimates (mean and standard deviation) for each biomarker.

    Args:
    data (pd.DataFrame): DataFrame containing participant data with columns 'participant', 
        'biomarker', 'measurement', and 'diseased'.
    # biomarkers (List[str]): A list of biomarker names.

    Returns:
    Dict[str, Dict[str, float]]: A dictionary where each key is a biomarker name,
        and each value is another dictionary containing the means and standard deviations 
        for theta and phi of that biomarker, with keys 'theta_mean', 'theta_std', 'phi_mean', 
        and 'phi_std'.
    """
    # empty list of dictionaries to store the estimates
    estimates = {}
    biomarkers = data.biomarker.unique()
    for biomarker in biomarkers:
        # Filter data for the current biomarker
        # reset_index is necessary here because we will use healthy_df.index later
        biomarker_df = data[data['biomarker']
                            == biomarker].reset_index(drop=True)
        theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_for_biomarker(
            biomarker_df)
        estimates[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std
        }
    return estimates


def fill_up_kj_and_affected(pdata, k_j):
    '''Fill up a single participant's data using k_j; basically add two columns: 
    k_j and affected
    Note that this function assumes that pdata already has the S_n column

    Input:
    - pdata: a dataframe of ten biomarker values for a specific participant 
    - k_j: a scalar
    '''
    data = pdata.copy()
    data['k_j'] = k_j
    data['affected'] = data.apply(lambda row: row.k_j >= row.S_n, axis=1)
    return data


def compute_single_measurement_likelihood(theta_phi, biomarker, affected, measurement):
    '''Computes the likelihood of the measurement value of a single biomarker

    We know the normal distribution defined by either theta or phi
    and we know the measurement. This will give us the probability
    of this given measurement value. 

    input:
    - theta_phi: the dictionary containing theta and phi values for each biomarker
    - biomarker: an integer between 0 and 9 
    - affected: boolean 
    - measurement: the observed value for a biomarker in a specific participant

    output: a scalar
    '''
    biomarker_dict = theta_phi[biomarker]
    mu = biomarker_dict['theta_mean'] if affected else biomarker_dict['phi_mean']
    std = biomarker_dict['theta_std'] if affected else biomarker_dict['phi_std']
    var = std**2
    if var <= int(0) or np.isnan(measurement) or np.isnan(mu):
        print(f"Invalid values: measurement: {measurement}, mu: {mu}, var: {var}")
        likelihood = np.exp(-(measurement - mu)**2 /
                            (2 * var)) / np.sqrt(2 * np.pi * var)
    else:
        likelihood = np.exp(-(measurement - mu)**2 /
                            (2 * var)) / np.sqrt(2 * np.pi * var)
    return likelihood

def compute_likelihood(pdata, k_j, theta_phi):
    '''This implementes the formula of https://ebm-book2.vercel.app/distributions.html#known-k-j
    This function computes the likelihood of seeing this sequence of biomarker values 
    for a specific participant, assuming that this participant is at stage k_j
    '''
    data = fill_up_kj_and_affected(pdata, k_j)
    likelihood = 1
    for i, row in data.iterrows():
        biomarker = row['biomarker']
        measurement = row['measurement']
        affected = row['affected']
        likelihood *= compute_single_measurement_likelihood(
            theta_phi, biomarker, affected, measurement)
    return likelihood


def shuffle_order(arr: np.ndarray, n_shuffle: int) -> None:
    """
    Randomly shuffle a specified number of elements in an array.

    Args:
    arr (np.ndarray): The array to shuffle elements in.
    n_shuffle (int): The number of elements to shuffle within the array.
    """
    if n_shuffle > len(arr):
        raise ValueError(
            "n_shuffle cannot be greater than the length of the array")

    # Randomly choose indices to shuffle
    indices = np.random.choice(len(arr), size=n_shuffle, replace=False)

    # Obtain and shuffle the elements at these indices
    selected_elements = arr[indices]
    np.random.shuffle(selected_elements)

    # Place the shuffled elements back into the array
    arr[indices] = selected_elements


def obtain_most_likely_order_dic(all_current_accepted_order_dicts, burn_in, thining):
    """Obtain the most likely order based on all the accepted orders 
    Inputs:
        - all_current_accepted_order_dicts 
        - burn_in
        - thining
    Outputs:
        - a dictionary where key is biomarker and value is the most likely order for that biomarker
        Note that in this dic, the order follows the same order as in 
        data_we_have.biomarker.unique()
    """
    biomarker_stage_probability_df = get_biomarker_stage_probability(
        all_current_accepted_order_dicts, burn_in, thining)
    od = OrderedDict()
    assigned_stages = set()

    for i, biomarker in enumerate(biomarker_stage_probability_df.index):
        # probability array for that biomarker
        prob_arr = np.array(biomarker_stage_probability_df.iloc[i, :])

        # Sort indices of probabilities in descending order
        sorted_indices = np.argsort(prob_arr)[::-1] + 1

        for stage in sorted_indices:
            if stage not in assigned_stages:
                od[biomarker] = int(stage)
                assigned_stages.add(stage)
                break
        else:
            raise ValueError(
                f"Could not assign a unique stage for biomarker {biomarker}.")
    return od


def get_biomarker_stage_probability(all_current_accepted_order_dicts, burn_in, thining):
    """filter through all_dicts using burn_in and thining 
    and for each biomarker, get probability of being in each possible stage

    Input:
        - all_current_accepted_order_dicts 
        - burn_in
        - thinning
    Output:
        - dff: a pandas dataframe where index is biomarker name, each col is each stage
        and each cell is the probability of that biomarker indicating that stage

        Note that in dff, its index follows the same order as data_we_have.biomarker.unique()
    """
    df = pd.DataFrame(all_current_accepted_order_dicts)
    df = df[(df.index > burn_in) & (df.index % thining == 0)]
    # Create an empty list to hold dictionaries
    dict_list = []

    # biomarkers are in the same order as data_we_have.biomarker.unique()
    biomarkers = np.array(df.columns)

    # iterate through biomarkers
    for biomarker in biomarkers:
        dic = {"biomarker": biomarker}
        # get the frequency of biomarkers
        # value_counts will generate a Series where index is each cell's value
        # and the value is the frequency of that value
        stage_counts = df[biomarker].value_counts()
        # for each stage
        # note that df.shape[1] should be equal to num_biomarkers
        for i in range(1, df.shape[1] + 1):
            # get stage:prabability
            dic[i] = stage_counts.get(i, 0)/len(df)
        dict_list.append(dic)

    dff = pd.DataFrame(dict_list)
    dff.set_index(dff.columns[0], inplace=True)
    return dff


def save_heatmap(all_dicts, burn_in, thining, folder_name, file_name, title):
    # Check if the directory exists
    if not os.path.exists(folder_name):
        # Create the directory if it does not exist
        os.makedirs(folder_name)
    biomarker_stage_probability_df = get_biomarker_stage_probability(
        all_dicts, burn_in, thining)
    sns.heatmap(biomarker_stage_probability_df,
                annot=True, cmap="Greys", linewidths=.5,
                cbar_kws={'label': 'Probability'},
                fmt=".1f",
                # vmin=0, vmax=1,
                )
    plt.xlabel('Stage')
    plt.ylabel('Biomarker')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{file_name}.png")
    # plt.savefig(f'{file_name}.pdf')
    plt.close()
