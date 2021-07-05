import os
import pandas as pd
import numpy as np


def load_data():
    """
    :return:
    spike_clusters: list of neuron's ID in a single tetrode
    spike_times: list of index of timestamp for each activity
    To get specific time, do spike_times / fs
    """
    spike_clusters = np.load("spike_clusters.npy").reshape(-1)
    spike_times = np.load("spike_times.npy").reshape(-1)
    cluster_df = pd.read_csv("cluster_group.tsv", sep='\t')
    return spike_clusters, spike_times, cluster_df

def get_sample_index(trial_info, trial_name):

    """
    :param trial_name: The name of the trial to be get the start and end index
    :return: The starting and ending sample index for each trial
    """
    end_index = trial_info.loc["Cumulative Samples"][trial_name]
    start_index = trial_info.loc["Cumulative Samples"][trial_name] - trial_info.loc["Samples"][trial_name]
    return start_index, end_index

def check_cluster_type(cluster_df):
    """
    :param cluster_df: a dataframe contain two columns: "cluster_id", "group"
    :return: a list contain all neuron id which is not noise and not mua
    """
    good_cluster = cluster_df[(cluster_df["group"] != "noise") & (cluster_df["group"] != "mua")]
    return list(good_cluster["cluster_id"])

def get_split_index(spike_times, samples_index_list):
    samples_index_list_copy = samples_index_list.copy()
    splite_index = []
    splite_index.append(0)
    for index in range(len(spike_times) - 1):
        if (spike_times[index] - samples_index_list_copy[0]) * (spike_times[index + 1] - samples_index_list_copy[0]) <= 0:
            splite_index.append(index + 1)
            samples_index_list_copy.remove(samples_index_list_copy[0])
    splite_index.append(len(spike_times) -1)
    return splite_index


def split_trial(trial_info, spike_times, spike_clusters, split_index):
    """
    The trial information can be found trial_info column name
    Split two list into different trials
    return data structure is a dictionary with trial name as key, pairs of (spike_time, spike_index) as value
    """
    trial_dict = {}
    for trial_index in range(len(trial_info.columns)):
        trial_samples = list(zip(spike_times[split_index[trial_index]:split_index[trial_index + 1]],
                            spike_clusters[split_index[trial_index]:split_index[trial_index + 1]]))
        index = trial_info.columns[trial_index]
        trial_dict[index] = trial_samples
    return trial_dict

def unzip_trial_dict(trial_dict, trial_name):
    '''
    upzip the zipped pair in dictionary values
    '''
    unzipped_object = zip(*trial_dict[trial_name])
    unzipped_list = list(unzipped_object)
    spike_times = list(unzipped_list[0])
    spike_clusters = list(unzipped_list[1])
    return spike_times, spike_clusters


def trunk_samples(trial_info, spike_clusters, spike_times, trial_name):
    '''
    trunk the samples by removing their tails
    '''
    for sample_index in range(len(spike_times)):
        start_index, end_index = get_sample_index(trial_info, trial_name)
        if spike_times[sample_index] <= trial_info.loc["modified_samples"][trial_name] + start_index:
            cut_index = sample_index
    return spike_times[:cut_index], spike_clusters[:cut_index]


def get_matrix_structure(trial_info, spike_times, cluster_df, trial_name):
    fs = 30000
    step_size = 0.025*fs
    start_index, end_index = get_sample_index(trial_info, trial_name)
    sample_duration = trial_info.loc["modified_samples"][trial_name]
    end_trucked_index = start_index + sample_duration
    bins = np.arange(start = start_index, stop = end_trucked_index, step = step_size)
    NData = np.zeros([max(list(cluster_df["cluster_id"])) + 1, bins.shape[0] + 1])
    reminder = spike_times[-1] - bins[-1]
    return step_size, bins, NData, reminder


def get_single_filled_matrix(spike_times, spike_clusters, NData, step_size, bins):
    for index in range(len(spike_times)):
        neuron_id = spike_clusters[index]
        bin_index = int((spike_times[index] - bins[0]) / step_size)
        NData[neuron_id][bin_index] += 1
    return NData


def get_actmat_for_a_trial(trial_info, absolute_path, trial_name, project_file_name):
    os.chdir(absolute_path)
    spike_clusters, spike_times, cluster_df = load_data()
    samples_index_list = list(trial_info.loc["Cumulative Samples"])
    split_index = get_split_index(spike_times, samples_index_list)
    trial_dict = split_trial(trial_info, spike_times, spike_clusters, split_index)
    spike_times, spike_clusters = unzip_trial_dict(trial_dict, trial_name)
    spike_times, spike_clusters = trunk_samples(trial_info, spike_clusters, spike_times, trial_name)
    step_size, bins, NData, reminder = get_matrix_structure(trial_info, spike_times, cluster_df, trial_name)
    actmat = get_single_filled_matrix(spike_times, spike_clusters, NData, step_size, bins)
    neuron_id = []
    for row_index in range(len(actmat)):
        if row_index in check_cluster_type(cluster_df):
            neuron_id.append(project_file_name + "_T" + absolute_path[absolute_path.index('Tetrode_') + 8 :
                                                                                (absolute_path.index('phy_') - 1)]
                             + '_UID' + str(row_index))
        else:
            neuron_id.append("null")
    actmat[:, len(actmat[0]) - 2] = actmat[:, len(actmat[0]) - 2] + actmat[:, len(actmat[0]) - 1]
    actmat = actmat[:, : len(actmat[0]) - 1]
    return actmat, neuron_id

def iterate_all_file_for_a_trial(trial_info, absolute_path_list, trial_name, project_file_name):
    # neuron_id_all = []
    actmat, neuron_id = get_actmat_for_a_trial(trial_info, absolute_path_list[0], trial_name, project_file_name)
    # neuron_id_all.append(neuron_id)
    for path_index in range(1, len(absolute_path_list)):
        matrix, neuron_id = get_actmat_for_a_trial(trial_info, absolute_path_list[path_index], trial_name, project_file_name)
        actmat = np.concatenate((actmat, matrix), axis=0)
        # neuron_id_all.append(neuron_id)
    # neuron_name = [item for sublist in neuron_id_all for item in sublist]
    return actmat
    # return actmat, neuron_name


def get_neuron_name(trial_info, absolute_path_list, trial_name, project_file_name):
    neuron_id_all = []
    actmat, neuron_id = get_actmat_for_a_trial(trial_info, absolute_path_list[0], trial_name, project_file_name)
    neuron_id_all.append(neuron_id)
    for path_index in range(1, len(absolute_path_list)):
        matrix, neuron_id = get_actmat_for_a_trial(trial_info, absolute_path_list[path_index], trial_name)
        neuron_id_all.append(neuron_id)
    neuron_name = [item for sublist in neuron_id_all for item in sublist]
    return neuron_name

def createFolder(absolute_dir):
    if not os.path.exists(absolute_dir):
        os.makedirs(absolute_dir)
        os.chdir(absolute_dir)
    else:
        os.chdir(absolute_dir)




