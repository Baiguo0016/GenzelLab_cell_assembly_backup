import os
import assembly as assembly
import numpy as np
import glob


def load_data():
    load_dict = {}
    for np_name in glob.glob('*.np[yz]'):
        load_dict[np_name[:-4]] = np.load(np_name)
    return load_dict

def clean_actmat(load_dict):
    del_indices = [i for i, x in enumerate(load_dict["actmat_neuron_name"]) if x == "null"]
    mat_dict = dict([(key, value) for key, value in load_dict.items() if '_actmat' in key.lower()])
    actmat_clean = {}
    for each_key, each_mat in mat_dict.items():
        new_mat = np.delete(each_mat,del_indices, 0)
        actmat_clean[each_key[:len(each_key) - 7]] = new_mat
    neuron_name =  dict(enumerate([x for x in load_dict["actmat_neuron_name"] if x != "null"]))
#     neuron_name =  [x for x in load_dict["Trial1_neuron_name"] if x != "null"]
    return actmat_clean, neuron_name

def get_interneurons_list(txt_file_name, project_name, condition_name, study_day_name, rat_name):
    inter_neurons = open(txt_file_name, "r")
    inter_neurons = list(inter_neurons.read().split(','))
    if '\n' in inter_neurons: inter_neurons.remove('\n')
    if '' in inter_neurons: inter_neurons.remove('')
    interneurons_list = []
    for interneuron in inter_neurons:
        study_day_interneuron = interneuron[interneuron.find("SD") : interneuron.find("_T")]
        # print(study_day_interneuron)
        study_day_match = study_day_interneuron == study_day_name
        # print(study_day_match)
        if (project_name in interneuron) and (condition_name in interneuron) and study_day_match and (rat_name in interneuron):
            interneurons_list.append(interneuron[: len(interneuron) - 3])
    return interneurons_list, inter_neurons

def find_index_interneurons(interneurons, name_dict):
    index_list = []
    for interneuron in interneurons:
        for  index, neuron_name in name_dict.items():
            y = neuron_name[neuron_name.find("_T") : ]
            x = interneuron[interneuron.find("_T") :]
#             print(x)
#             print(y)
            match = x == y
            if match:
                index_list.append(index)
    return index_list

def remove_interneuron(interneuron_index_list, neuron_name_dict, actmat_dict):
    load_dict = {}
    for key, actmat in actmat_dict.items():
        actmat_new = np.delete(actmat, interneuron_index_list, 0)
        load_dict[key] = actmat_new

    neuron_name_list = []
    for i in range(len(neuron_name_dict)):
        if i not in interneuron_index_list:
            neuron_name_list.append(neuron_name_dict[i])
    neuron_name_new =  dict(enumerate([x for x in neuron_name_list]))
#     print(neuron_name_new)
    return load_dict, neuron_name_new
