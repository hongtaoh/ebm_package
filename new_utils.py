from typing import List, Optional
import json 
import pandas as pd 
import numpy as np 
import os 
import scipy.stats as stats
from scipy.stats import kendalltau
from typing import List, Optional, Tuple, Dict
from scipy.stats import mode
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import math 

def generate_data_from_ebm(
    n_participants: int,
    S_ordering: List[str],
    real_theta_phi_file: str,
    healthy_ratio: float,
    output_dir: str,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:
    n_participants (int): Number of participants.
    S_ordering (List[str]): Biomarker names ordered according to the order 
        in which each of them get affected by the disease.
    real_theta_phi_file (str): Directory of a JSON file which contains 
        theta and phi values for all biomarkers.
        See real_theta_phi.json for example format.
    output_dir (str): Directory where output files will be saved.
    healthy_ratio (float): Proportion of healthy participants out of n_participants.
    seed (Optional[int]): Seed for the random number generator for reproducibility.

    Returns:
    pd.DataFrame: A DataFrame with columns 'participant', "biomarker", 'measurement', 
        'diseased'.
    """
    # Parameter validation
    assert n_participants > 0, "Number of participants must be greater than 0."
    assert 0 <= healthy_ratio <= 1, "Healthy ratio must be between 0 and 1."

    # Set the seed for numpy's random number generator
    rng = np.random.default_rng(seed)

    # Load theta and phi values from the JSON file
    try:
        with open(real_theta_phi_file) as f:
            real_theta_phi = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {real_theta_phi} not fount")
    except json.JSONDecodeError:
        raise ValueError(f"File {real_theta_phi_file} is not a valid JSON file.")

    n_biomarkers = len(S_ordering)
    n_stages = n_biomarkers + 1

    n_healthy = int(n_participants * healthy_ratio)
    n_diseased = int(n_participants - n_healthy)

    # Generate disease stages
    kjs = np.concatenate((np.zeros(n_healthy, dtype=int), rng.integers(1, n_stages, n_diseased)))
    # shuffle so that it's not 0s first and then disease stages bur all random
    rng.shuffle(kjs)

    # Initiate biomarker measurement matrix (J participants x N biomarkers) with None
    X = np.full((n_participants, n_biomarkers), None, dtype=object)

    # Create distributions for each biomarker
    theta_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['theta_mean'],
        real_theta_phi[biomarker]['theta_std']
    ) for biomarker in S_ordering}

    phi_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['phi_mean'],
        real_theta_phi[biomarker]['phi_std']
    ) for biomarker in S_ordering}

    # Populate the matrix with biomarker measurements
    for j in range(n_participants):
        for n, biomarker in enumerate(S_ordering):
            k_j = kjs[j]
            S_n = n + 1

            # Assign biomarker values based on the participant's disease stage
            # affected, or not_affected, is regarding the biomarker, not the participant
            if k_j >= 1:
                if k_j >= S_n:
                    # rvs() is affected by np.random()
                    X[j, n] = (
                        j, biomarker, theta_dist[biomarker].rvs(random_state=rng), k_j, S_n, 'affected')
                else:
                    X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                               k_j, S_n, 'not_affected')
            # if the participant is healthy
            else:
                X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(random_state=rng),
                           k_j, S_n, 'not_affected')

    df = pd.DataFrame(X, columns=S_ordering)
    # make this dataframe wide to long
    df_long = df.melt(var_name="Biomarker", value_name="Value")
    data = df_long['Value'].apply(pd.Series)
    data.columns = ['participant', "biomarker", 'measurement', 'k_j', 'S_n', 'affected_or_not']

    biomarker_name_change_dic = dict(zip(S_ordering, range(1, n_biomarkers + 1)))
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    data.drop(['k_j', 'S_n', 'affected_or_not'], axis=1, inplace=True)
    data['biomarker'] = data.apply(
        lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis=1)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    combination_str = f"{int(healthy_ratio*n_participants)}|{n_participants}"
    data.to_csv(f'{output_dir}/{combination_str}.csv', index=False)
    print("Data generation done! Output saved to:", combination_str)
    return data

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
    clustering_setup = KMeans(n_clusters=2, random_state=0, n_init="auto")

    # you need to make sure each measurment is a np.array before putting it into "fit"
    measurements = np.array(biomarker_df['measurement']).reshape(-1, 1)

    # Fit clustering method
    clustering_result = clustering_setup.fit(measurements)
    predictions = clustering_result.labels_

    # dataframe for non-diseased participants
    healthy_df = biomarker_df[biomarker_df['diseased'] == False]
    healthy_measurements = np.array(healthy_df['measurement']).reshape(-1, 1)
    # which cluster are healthy participants in
    healthy_predictions = predictions[healthy_df.index]

    # the mode of the above predictions will be the phi cluster index
    phi_cluster_idx = mode(healthy_predictions, keepdims=False).mode
    theta_cluster_idx = 1 - phi_cluster_idx

    if len(set(healthy_predictions)) > 1:
        # Reassign clusters using Agglomerative Clustering
        clustering = AgglomerativeClustering(
            n_clusters=2).fit(healthy_measurements)

        # Find the dominant cluster for healthy participants
        phi_cluster_idx = mode(clustering.labels_, keepdims=False).mode

        # Update predictions to ensure all healthy participants are in the dominant cluster
        updated_predictions = predictions.copy()
        for i in healthy_df.index:
            updated_predictions[i] = phi_cluster_idx
    else:
        updated_predictions = predictions

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
    return theta_mean, theta_std, phi_mean, phi_std

def get_theta_phi_estimates(
        data: pd.DataFrame,
        biomarkers: List[str],
    ) -> Dict[str, Dict[str, float]]:
    """
    Obtain theta and phi estimates (mean and standard deviation) for each biomarker.

    Args:
    data (pd.DataFrame): DataFrame containing participant data with columns 'participant', 
        'biomarker', 'measurement', and 'diseased'.
    biomarkers (List[str]): A list of biomarker names.

    Returns:
    Dict[str, Dict[str, float]]: A dictionary where each key is a biomarker name,
        and each value is another dictionary containing the means and standard deviations 
        for theta and phi of that biomarker, with keys 'theta_mean', 'theta_std', 'phi_mean', 
        and 'phi_std'.
    """
    # empty list of dictionaries to store the estimates
    estimates = {}
    for biomarker in biomarkers:
        # Filter data for the current biomarker
        # reset_index is necessary here because we will use healthy_df.index later
        biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
        theta_mean, theta_std, phi_mean, phi_std = compute_theta_phi_for_biomarker(biomarker_df)
        estimates[biomarker] = {
            'theta_mean': theta_mean,
            'theta_std': theta_std,
            'phi_mean': phi_mean,
            'phi_std': phi_std
        }
    return estimates

def fill_up_pdata(pdata, k_j):
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
    likelihood = np.exp(-(measurement - mu)**2/(2*var))/np.sqrt(2*np.pi*var)
    return likelihood

def compute_likelihood(pdata, k_j, theta_phi):
    '''This implementes the formula of https://ebm-book2.vercel.app/distributions.html#known-k-j
    This function computes the likelihood of seeing this sequence of biomarker values 
    for a specific participant, assuming that this participant is at stage k_j
    '''
    data = fill_up_pdata(pdata, k_j)
    likelihood = 1
    for i, row in data.iterrows():
        biomarker = row['biomarker']
        measurement = row['measurement']
        affected = row['affected']
        likelihood *= compute_single_measurement_likelihood(
            theta_phi, biomarker, affected, measurement)
    return likelihood

def calculate_soft_kmeans_for_biomarker(
        data,
        biomarker,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None
):
    """
    Calculate mean and std for both the affected and non-affected clusters for a single biomarker.

    Parameters:
        data (pd.DataFrame): The data containing measurements.
        biomarker (str): The biomarker to process.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of 
            dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Means and standard deviations for affected and non-affected clusters.
    """
    if seed is not None:
        # Set the seed for numpy's random number generator
        rng = np.random.default_rng(seed)

    # DataFrame for this biomarker
    biomarker_df = data[data['biomarker'] == biomarker].reset_index(drop=True)
    # Extract measurements
    measurements = np.array(biomarker_df['measurement'])

    this_biomarker_order = order_dict[biomarker]

    affected_cluster = []
    non_affected_cluster = []

    for p in range(n_participants):
        if p in non_diseased_participants:
            non_affected_cluster.append(measurements[p])
        else:
            if this_biomarker_order == 1:
                affected_cluster.append(measurements[p])
            else:
                normalized_stage_likelihood_dict = hashmap_of_normalized_stage_likelihood_dicts[
                    p]
                # Calculate probabilities for affected and non-affected states
                affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s >= this_biomarker_order
                )
                non_affected_prob = sum(
                    normalized_stage_likelihood_dict[s] for s in diseased_stages if s < this_biomarker_order
                )
                if affected_prob > non_affected_prob:
                    affected_cluster.append(measurements[p])
                elif affected_prob < non_affected_prob:
                    non_affected_cluster.append(measurements[p])
                else:
                    # Assign to either cluster randomly if probabilities are equal
                    if rng.random() > 0.5:
                        affected_cluster.append(measurements[p])
                    else:
                        non_affected_cluster.append(measurements[p])

    # Compute means and standard deviations
    theta_mean = np.mean(affected_cluster) if affected_cluster else np.nan
    theta_std = np.std(affected_cluster) if affected_cluster else np.nan
    phi_mean = np.mean(
        non_affected_cluster) if non_affected_cluster else np.nan
    phi_std = np.std(non_affected_cluster) if non_affected_cluster else np.nan

    return theta_mean, theta_std, phi_mean, phi_std

def soft_kmeans_theta_phi_estimates(
        iteration,
        prior_theta_phi_estimates,
        data_we_have,
        biomarkers,
        order_dict,
        n_participants,
        non_diseased_participants,
        hashmap_of_normalized_stage_likelihood_dicts,
        diseased_stages,
        seed=None):
    """
    Get the DataFrame of theta and phi using the soft K-means algorithm for all biomarkers.

    Parameters:
        data_we_have (pd.DataFrame): DataFrame containing the data.
        biomarkers (list): List of biomarkers in string.
        order_dict (dict): Dictionary mapping biomarkers to their order.
        n_participants (int): Number of participants in the study.
        non_diseased_participants (list): List of non-diseased participants.
        hashmap_of_normalized_stage_likelihood_dicts (dict): Hash map of dictionaries containing stage likelihoods for each participant.
        diseased_stages (list): List of diseased stages.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        a dictionary containing the means and standard deviations for theta and phi for each biomarker.
    """
    # List of dicts to store the estimates
    hashmap_of_means_stds_estimate_dicts = {}
    for biomarker in biomarkers:
        dic = {'biomarker': biomarker}
        prior_theta_phi_estimates_biomarker = prior_theta_phi_estimates[biomarker]
        theta_mean, theta_std, phi_mean, phi_std = calculate_soft_kmeans_for_biomarker(
            data_we_have,
            biomarker,
            order_dict,
            n_participants,
            non_diseased_participants,
            hashmap_of_normalized_stage_likelihood_dicts,
            diseased_stages,
            seed
        )
        if theta_std == 0 or math.isnan(theta_std):
            theta_mean = prior_theta_phi_estimates_biomarker['theta_mean']
            theta_std = prior_theta_phi_estimates_biomarker['theta_std']
        if phi_std == 0 or math.isnan(phi_std):
            phi_mean = prior_theta_phi_estimates_biomarker['phi_mean']
            phi_std = prior_theta_phi_estimates_biomarker['phi_std']
        dic['theta_mean'] = theta_mean
        dic['theta_std'] = theta_std
        dic['phi_mean'] = phi_mean
        dic['phi_std'] = phi_std
        hashmap_of_means_stds_estimate_dicts[biomarker] = dic
    return hashmap_of_means_stds_estimate_dicts

"""
If soft kmeans, no matter uniform prior on kjs or not, I always need to update hashmap of dicts
    This is because, even if when we do not have uniform prior, we don't need normalized_stage_likelihood_dict
    to calculate the weighted average, we still need it to calculate soft kmeans
If kmeans only, if with uniform prior, we don't need normalized_stage_likelihood_dict to calculate 
    weighted average;
    but we do need to calculate normalized_stage_likelihood_dict when without uniform prior 
"""

def calculate_all_participant_ln_likelihood_and_update_hashmap(
        iteration,
        data_we_have,
        current_order_dict,
        n_participants,
        non_diseased_participant_ids,
        theta_phi_estimates,
        diseased_stages,
):
    data = data_we_have.copy()
    data['S_n'] = data.apply(
        lambda row: current_order_dict[row['biomarker']], axis=1)
    all_participant_ln_likelihood = 0
    # key is participant id
    # value is normalized_stage_likelihood_dict
    hashmap_of_normalized_stage_likelihood_dicts = {}
    for p in range(n_participants):
        pdata = data[data.participant == p].reset_index(drop=True)
        if p in non_diseased_participant_ids:
            this_participant_likelihood = compute_likelihood(
                pdata, k_j=0, theta_phi=theta_phi_estimates)
            this_participant_ln_likelihood = np.log(
                this_participant_likelihood + 1e-10)
        else:
            normalized_stage_likelihood_dict = None
            # initiaze stage_likelihood
            stage_likelihood_dict = {}
            for k_j in diseased_stages:
                kj_likelihood = compute_likelihood(
                    pdata, k_j, theta_phi_estimates)
                # update each stage likelihood for this participant
                stage_likelihood_dict[k_j] = kj_likelihood
            # Add a small epsilon to avoid division by zero
            likelihood_sum = sum(stage_likelihood_dict.values())
            epsilon = 1e-10
            if likelihood_sum == 0:
                print("Invalid likelihood_sum: zero encountered.")
                likelihood_sum = epsilon  # Handle the case accordingly
            normalized_stage_likelihood = [
                l/likelihood_sum for l in stage_likelihood_dict.values()]
            normalized_stage_likelihood_dict = dict(
                zip(diseased_stages, normalized_stage_likelihood))
            hashmap_of_normalized_stage_likelihood_dicts[p] = normalized_stage_likelihood_dict

            # calculate weighted average
            this_participant_likelihood = np.mean(likelihood_sum)
            this_participant_ln_likelihood = np.log(this_participant_likelihood + 1e-10)
        all_participant_ln_likelihood += this_participant_ln_likelihood
    return all_participant_ln_likelihood, hashmap_of_normalized_stage_likelihood_dicts


def shuffle_order(arr: np.ndarray, n_shuffle: int) -> None:
    """
    Randomly shuffle a specified number of elements in an array.

    Args:
    arr (np.ndarray): The array to shuffle elements in.
    n_shuffle (int): The number of elements to shuffle within the array.
    """
    if n_shuffle > len(arr):
        raise ValueError("n_shuffle cannot be greater than the length of the array")

    # Randomly choose indices to shuffle
    indices = np.random.choice(len(arr), size=n_shuffle, replace=False)
    
    # Obtain and shuffle the elements at these indices
    selected_elements = arr[indices]
    np.random.shuffle(selected_elements)
    
    # Place the shuffled elements back into the array
    arr[indices] = selected_elements

def metropolis_hastings_soft_kmeans(
    data_we_have,
    iterations,
    n_shuffle,
    log_folder_name,
):
    '''Implement the metropolis-hastings algorithm
    Inputs: 
        - data: data_we_have
        - iterations: number of iterations
        - log_folder_name: the folder where log files locate

    Outputs:
        - best_order: a numpy array
        - best_likelihood: a scalar 
    '''
    n_participants = len(data_we_have.participant.unique())
    biomarkers = data_we_have.biomarker.unique()
    n_biomarkers = len(biomarkers)
    n_stages = n_biomarkers + 1
    non_diseased_participant_ids = data_we_have.loc[
        data_we_have.diseased == False].participant.unique()
    diseased_stages = np.arange(start=1, stop=n_stages, step=1)
    # obtain the iniial theta and phi estimates
    prior_theta_phi_estimates = get_theta_phi_estimates(
        data_we_have,
        biomarkers,
    )
    theta_phi_estimates = prior_theta_phi_estimates.copy()

    # initialize empty lists
    all_order_dicts = []
    all_current_accepted_likelihoods = []
    acceptance_count = 0
    all_current_acceptance_ratios = []
    all_current_accepted_order_dicts = []
    terminal_output_strings = []
    hashmaps_of_theta_phi_estimates = {}
    hashmap_of_estimated_theta_phi_dicts = {}

    current_accepted_order = np.random.permutation(np.arange(1, n_stages))
    current_accepted_order_dict = dict(zip(biomarkers, current_accepted_order))
    current_accepted_likelihood = -np.inf

    for _ in range(iterations):
        new_order = current_accepted_order.copy()
        # random.shuffle(new_order)
        shuffle_order(new_order, n_shuffle)
        current_order_dict = dict(zip(biomarkers, new_order))
        all_participant_ln_likelihood, \
            hashmap_of_normalized_stage_likelihood_dicts = calculate_all_participant_ln_likelihood_and_update_hashmap(
                _,
                data_we_have,
                current_order_dict,
                n_participants,
                non_diseased_participant_ids,
                theta_phi_estimates,
                diseased_stages,
            )

        # Now, update theta_phi_estimates using soft kmeans
        # based on the updated hashmap of normalized stage likelihood dicts
        theta_phi_estimates = soft_kmeans_theta_phi_estimates(
            _,
            prior_theta_phi_estimates,
            data_we_have,
            biomarkers,
            current_order_dict,
            n_participants,
            non_diseased_participant_ids,
            hashmap_of_normalized_stage_likelihood_dicts,
            diseased_stages,
            seed=None,
        )

        # Log-Sum-Exp Trick
        max_likelihood = max(all_participant_ln_likelihood,
                             current_accepted_likelihood)
        prob_of_accepting_new_order = np.exp(
            (all_participant_ln_likelihood - max_likelihood) -
            (current_accepted_likelihood - max_likelihood)
        )

        # prob_of_accepting_new_order = np.exp(
        #     all_participant_ln_likelihood - current_accepted_likelihood)

        # it will definitly update at the first iteration
        if np.random.rand() < prob_of_accepting_new_order:
            acceptance_count += 1
            current_accepted_order = new_order
            current_accepted_likelihood = all_participant_ln_likelihood
            current_accepted_order_dict = current_order_dict

        all_current_accepted_likelihoods.append(current_accepted_likelihood)
        acceptance_ratio = acceptance_count*100/(_+1)
        all_current_acceptance_ratios.append(acceptance_ratio)
        all_order_dicts.append(current_order_dict)
        all_current_accepted_order_dicts.append(current_accepted_order_dict)
        hashmaps_of_theta_phi_estimates[_] = theta_phi_estimates
        # update theta_phi_dic
        hashmap_of_estimated_theta_phi_dicts[_] = theta_phi_estimates

        if (_+1) % 10 == 0:
            formatted_string = (
                f"iteration {_ + 1} done, "
                f"current accepted likelihood: {current_accepted_likelihood}, "
                f"current acceptance ratio is {acceptance_ratio:.2f} %, "
                f"current accepted order is {current_accepted_order_dict}, "
            )
            terminal_output_strings.append(formatted_string)
            print(formatted_string)

    final_acceptance_ratio = acceptance_count/iterations

    save_output_strings(log_folder_name, terminal_output_strings)

    save_all_dicts(all_order_dicts, log_folder_name, "all_order")
    save_all_dicts(
        all_current_accepted_order_dicts,
        log_folder_name,
        "all_current_accepted_order_dicts")
    save_all_current_accepted(
        all_current_accepted_likelihoods,
        "all_current_accepted_likelihoods",
        log_folder_name)
    save_all_current_accepted(
        all_current_acceptance_ratios,
        "all_current_acceptance_ratios",
        log_folder_name)
    # save hashmap_of_estimated_theta_and_phi_dicts
    with open(f'{log_folder_name}/hashmap_of_estimated_theta_phi_dicts.json', 'w') as fp:
        json.dump(hashmap_of_estimated_theta_phi_dicts, fp)
    print("done!")
    return (
        current_accepted_order_dict,
        all_order_dicts,
        all_current_accepted_order_dicts,
        all_current_accepted_likelihoods,
        all_current_acceptance_ratios,
        final_acceptance_ratio,
        hashmap_of_estimated_theta_phi_dicts,
    )

def save_output_strings(
        log_folder_name,
        terminal_output_strings
):
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)
    terminal_output_filename = f"{log_folder_name}/terminal_output.txt"
    with open(terminal_output_filename, 'w') as file:
        for result in terminal_output_strings:
            file.write(result + '\n')

def save_all_dicts(all_dicts, log_folder_name, file_name):
    """Save all_dicts into a CSV file within a specified directory.

    If the directory does not exist, it will be created.
    """
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_dicts)

    # Add an 'iteration' column
    df['iteration'] = np.arange(start=1, stop=len(df) + 1, step=1)

    # Set 'iteration' as the index
    df.set_index("iteration", inplace=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f"{log_folder_name}/{file_name}.csv", index=True)


def save_all_current_accepted(var, var_name, log_folder_name):
    """save all_current_order_dicts, all_current_ikelihoods, 
    and all_current_acceptance_ratios
    """
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)
    x = np.arange(start=1, stop=len(var) + 1, step=1)
    df = pd.DataFrame({"iteration": x, var_name: var})
    df = df.set_index('iteration')
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=True)


def save_all_current_participant_stages(var, var_name, log_folder_name):
    # Check if the directory exists
    if not os.path.exists(log_folder_name):
        # Create the directory if it does not exist
        os.makedirs(log_folder_name)
    df = pd.DataFrame(var)
    df.index.name = 'iteration'
    df.index = df.index + 1
    df.to_csv(f"{log_folder_name}/{var_name}.csv", index=False)