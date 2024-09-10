import scipy.stats as stats
import json 
import numpy as np
import pandas as pd 

def generate_data_from_ebm(
    n_participants,
    S_ordering,
    real_theta_phi_file,
    healthy_ratio,
    output_dir,
    seed=None  # Add seed parameter
):
    """
    Simulate an Event-Based Model (EBM) for disease progression.

    Args:
    n_participants (int): Number of participants.
    S_ordering (List): Biomarker names ordered according to the order 
        in which each of them get affected by the disease
    real_theta_phi (JSON file):  theta and phi values for all biomarkers
        See real_theta_phi.json for example format
    output_dir (Str): directory where output files will be saved
    healthy_ratio: # of healthy participants/n_participants

    Returns:
    pandas dataframe: colmns are 'participant', "biomarker", 'measurement', 
        'k_j', 'S_n', 'affected_or_not'
    """
    if seed is not None:
        # Set the seed for numpy's random number generator
        np.random.seed(seed)

    with open(real_theta_phi_file) as f:
        real_theta_phi = json.load(f)
    # biomarkers = np.array(real_theta_phi.biomarker)
    n_biomarkers = len(S_ordering)
    n_stages = n_biomarkers + 1

    n_healthy = int(n_participants*healthy_ratio)
    n_diseased = int(n_participants - n_healthy)

    # generate kjs
    zeros = np.zeros(n_healthy, dtype=int)
    random_integers = np.random.randint(1, n_stages, n_diseased)
    kjs = np.concatenate((zeros, random_integers))
    np.random.shuffle(kjs)

    # Initiate biomarker measurement matrix (J participants x N biomarkers),
    # with entries as None
    X = np.full((n_participants, n_biomarkers), None, dtype=object)

    # biomarker : normal distribution
    theta_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['theta_mean'],
        real_theta_phi[biomarker]['theta_std']
    ) for biomarker in S_ordering}
    phi_dist = {biomarker: stats.norm(
        real_theta_phi[biomarker]['phi_mean'],
        real_theta_phi[biomarker]['phi_std']
    ) for biomarker in S_ordering}

    # Iterate through participants
    for j in range(n_participants):
        # Iterate through biomarkers
        for n, biomarker in enumerate(S_ordering):
            # Disease stage of the current participant
            k_j = kjs[j]
            # Disease stage indicated by the current biomarker
            # Note that biomarkers always indicate that the participant is diseased
            # Thus, S_n >= 1
            S_n = np.where(S_ordering == biomarker)[0][0] + 1

            # Assign biomarker values based on the participant's disease stage
            # affected, or not_affected, is regarding the biomarker, not the participant
            if k_j >= 1:
                if k_j >= S_n:
                    # rvs() is affected by np.random()
                    X[j, n] = (
                        j, biomarker, theta_dist[biomarker].rvs(), k_j, S_n, 'affected')
                else:
                    X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(),
                               k_j, S_n, 'not_affected')
            # if the participant is healthy
            else:
                X[j, n] = (j, biomarker, phi_dist[biomarker].rvs(),
                           k_j, S_n, 'not_affected')

    df = pd.DataFrame(X, columns=S_ordering)
    # make this dataframe wide to long
    df_long = df.melt(var_name="Biomarker", value_name="Value")
    data = df_long['Value'].apply(pd.Series)
    data.columns = [
        'participant',
        "biomarker",
        'measurement',
        'k_j',
        'S_n',
        'affected_or_not'
    ]
    # values_df.to_csv("data/participant_data.csv", index=False)
    biomarker_name_change_dic = dict(
        zip(S_ordering, range(1, n_biomarkers + 1)))
    data['diseased'] = data.apply(lambda row: row.k_j > 0, axis=1)
    data = data.drop(['k_j', 'S_n', 'affected_or_not'], axis=1)
    data['biomarker'] = data.apply(
        lambda row: f"{row.biomarker} ({biomarker_name_change_dic[row.biomarker]})", axis=1)
    if not os.path.exists("data/synthetic"):
        # Create the directory if it does not exist
        os.makedirs("data/synthetic")
    comb_str = f"{int(healthy_ratio*n_participants)}|{n_participants}"
    data.to_csv(f'data/synthetic/{comb_str}.csv', index=False)
    print("data generation done!")
    return data