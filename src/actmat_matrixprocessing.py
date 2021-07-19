import os
import assembly as assembly
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from bokeh.plotting import figure
from bokeh.io import push_notebook, show, output_notebook
import plotly.graph_objects as go
import pickle

def load_actmat(path):
    os.chdir(path)
    neuron_name_file = open("neuron_name_dict.pkl", "rb")
    neuron_name_dict = pickle.load(neuron_name_file)
    actmat_file = open("actmat_dict.pkl", "rb")
    actmat_dict = pickle.load(actmat_file)
    return neuron_name_dict, actmat_dict


def combine_all_trial(actmat_dict, ordered_list):
    time_all = np.hstack((actmat_dict[ordered_list[0]], actmat_dict[ordered_list[1]]))
    for item in ordered_list[2:]:
        time_all = np.hstack((time_all, actmat_dict[item]))
    return time_all


def get_actmat_dict_with_split_trial5(actmat_dict):
    trial5_actmat = actmat_dict['Post_Trial5']
    trial5_three_actmat = np.hsplit(trial5_actmat, 4)
#     print(trial5_three_actmat[0])
    actmat_dict['PT5_part1'] = trial5_three_actmat[0]
    actmat_dict['PT5_part2'] = trial5_three_actmat[1]
    actmat_dict['PT5_part3'] = trial5_three_actmat[2]
    actmat_dict['PT5_part4'] = trial5_three_actmat[3]
    actmat_dict.pop('Post_Trial5')
    return actmat_dict

def get_important_neuron(neuron_name_dict, patterns, assembly_group):
    important_neuron = []
    for neuron_index in range(len(patterns[assembly_group])):
        if abs(patterns[assembly_group][neuron_index]) > 2 * np.std(patterns[assembly_group]):
            important_neuron.append(neuron_name_dict[neuron_index] + " with index" + " " + str(neuron_index))
    return important_neuron


def get_important_neuron_by_threshold(neuron_name_dict, patterns, assembly_group, threshold):
    important_neuron = []
    for neuron_index in range(len(patterns[assembly_group])):
        if patterns[assembly_group][neuron_index] > threshold:
            important_neuron.append(neuron_name_dict[neuron_index] + " with index" + " " + str(neuron_index))
    return important_neuron


def plot_actmat_assembly_act(zactmat_variable, assemblyAct_matrix, xrange):
    plt.figure(figsize=(15,6))
    s1 = plt.subplot(211)
    plt.imshow(zactmat_variable[:, :xrange],cmap='Reds',interpolation='nearest',aspect='auto')
    plt.title('z-scored activity matrix')
    plt.ylabel('neuron #')

    plt.subplot(212,sharex=s1)
#     plt.plot(assemblyAct_matrix[:, 0 : xrange].T,linewidth=4)
    for index in range(len(assemblyAct_matrix)):
        label_str = 'index'+ str(index)
        plt.plot(assemblyAct_matrix[index], label = label_str)
    plt.title('assembly activities')
    plt.xlim(0, xrange)
    plt.xlabel('time bin')
    plt.ylabel('strength')
    plt.tight_layout()
    plt.legend()



def show_zactmat(zactmat_matrix):
    TOOLTIPS = [
        ("(x,y)", "($x, $y)")
    ]

    p = figure(plot_width=800, plot_height=400, tooltips=TOOLTIPS)
    p.x_range.range_padding = p.y_range.range_padding = 0
    p.image(image=[zactmat_matrix], x=0, y=0, dw=np.shape(zactmat_matrix)[1], dh=np.shape(zactmat_matrix)[0], palette="Spectral11", level="image")
    p.xaxis.axis_label = 'time bins'
    p.yaxis.axis_label = 'neuron #'
    show(p)

def plot_zactmat(zactmat_variable, xrange):
    plt.imshow(zactmat_variable[:, :xrange],cmap='Reds',interpolation='nearest',aspect='auto')
    plt.title('z-scored activity matrix')
    plt.ylabel('neuron #')


def plot_assembly_act(assemblyAct_matrix, xrange):
    plt.figure(figsize=(15,10))
    for index in range(len(assemblyAct_matrix)):
        label_str = 'index'+ str(index)
        plt.plot(assemblyAct_matrix[index], label = label_str)
    plt.title('assembly activities')
    plt.xlim(0, xrange)
    plt.xlabel('time bin')
    plt.ylabel('strength')
    plt.tight_layout()
    plt.legend()

def show_assemblyActivity(assemblyAct, ending_time, title_string):
    N = np.shape(assemblyAct)[1]
    x = np.linspace(0, ending_time, N)
    fig = go.Figure()

    for index in range(len(assemblyAct)):
        new_trace = assemblyAct[index]
        fig.add_trace(go.Scatter(x=x, y=new_trace,
                        mode='lines',
                        name='assembly ' + str(index)))

    fig.update_layout(
    title= title_string,
    xaxis_title="time (minutes)",
    yaxis_title="Strength",
    legend_title="assemblies",
)
    fig.write_html(title_string + ".html")
    fig.show()


def get_zscore(actmat):
    silentneurons = np.var(actmat,axis=1)==0
    actmat_ = actmat[~silentneurons,:]
    # z-scoring activity matrix
    zactmat_ = stats.zscore(actmat_,axis=1)
    actmat[~silentneurons, :] = zactmat_
    return actmat


def clean_assemblyAct(assemblyAct, threshold = 5):
    new_assembly = assemblyAct.copy()
    new_assembly[new_assembly <= threshold] = 0
    return new_assembly


def process_pattern(patterns):
    for index in range(len(patterns)):
        if abs(patterns[index].min()) > abs(patterns[index].max()):
            patterns[index] = -patterns[index]
    return patterns


def get_save_important_neuron(neuron_name_dict, patterns):
    important_neuron_dict = {}
    for assembly_index in range(len(patterns)):
        important_neuron_dict[assembly_index] = get_important_neuron(neuron_name_dict, patterns, assembly_index)

    important_neuron_file = open("important_neuron_per_assembly.pkl", "wb")
    pickle.dump(important_neuron_dict, important_neuron_file)
    important_neuron_file.close()
    return important_neuron_dict





