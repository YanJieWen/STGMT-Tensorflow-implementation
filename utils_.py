# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : utils_
# @Project Name :metacode
from copyreg import pickle
from Hyperparameters import Hyperparameters as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import pickle
#building a graph
def build_graph(path_file):
    """

    :param path_file: the adj matrix file path
    :return: a graph with shape with the graph informations
    """
    df = pd.read_csv(path_file)
    graph = nx.DiGraph()#a garph with direction
    edges = df.values
    graph.add_weighted_edges_from(edges)
    #visualiazation
    pos=nx.random_layout(graph)                     # gen the location of nodes
    plt.rcParams['figure.figsize']= (6, 4)      # picture size
    nx.draw(graph,pos,with_labels=True, node_color='white', edge_color='red', node_size=15, alpha=0.5 )
    plt.title('Self_Define Net',fontsize=18)
    plt.show()
    return graph

def toarray(data):#get array
    if isinstance(data,list):
        return np.array(data)
    else:
        return data

def minmaxsca(datas):
    """

    :param datas: (points,sensors,flow)
    :return:(points,sensors,flow),a log
    """
    n,s,f = datas.shape
    sca = MinMaxScaler(feature_range=(-1,1))
    datas = sca.fit_transform(datas.reshape(-1,f)).reshape(-1,s,f)
    return datas,sca
def inverse_minmaxsca(datas,sca):
    """

    :param datas: (points,sensors,flow) after minmax
    :param sca: a log feature
    :return: to inverse the datas to the original value data
    """
    n, s, f = datas.shape
    data = sca.inverse_transform(datas.reshape(-1,f)).reshape(-1,s,f)
    return data

def sim_distance(adj_matrix,df):
    """

    :param adj_matrix: (N,N)
    :param df: distance dataframe
    :return: (N,N)
    """
    W2 = adj_matrix*adj_matrix
    dis_sim = np.exp(-W2 / (np.std(df['cost']) * np.std(df['cost'])))
    dis_sim[np.where(dis_sim <= hp.eplision)] = 1#first set value==1 if <=eplision, then set 1==0 inverse select
    dis_sim[np.where(dis_sim == 1)] = 0
    return dis_sim

def time_sim_matrix(datas):
    """

    :param datas: origins data
    :return: (N,N)
    """
    num_sensors = datas.shape[1]
    sim_init = np.zeros((num_sensors, num_sensors))
    for i in range(num_sensors):
        for j in range(num_sensors):
            set_1 = datas[:288 * 7, i, :]#a week data, more data in not needed
            set_2 = datas[:288 * 7, j, :]
            time_sim = np.sum(set_1 * set_2) / (np.linalg.norm(set_1) * np.linalg.norm(set_2))
            sim_init[i, j] = time_sim
    sim_init[np.where(sim_init <= hp.time_eplision)] = 0
    diag = np.identity(sim_init.shape[0])#add self-loop
    sim_init+=diag
    x_norm = sim_init / np.sum(sim_init, axis=1,keepdims=True)
    return x_norm

def norm_adjmatrix(cal_matrix):
    """
    row norm for distsim_matrix
    :param cal_matrix: (N,N),after calculating the matrix
    :return: (N,N)
    """
    diag = np.identity(cal_matrix.shape[0])
    cal_matrix+=diag#add self-loop
    x_norm = cal_matrix/np.sum(cal_matrix,axis=1,keepdims=True)
    return x_norm

def read_pkl(data_path):
    with open(data_path,'rb') as f:
        return pickle.load(f)




















#并不能根据节点流量统计出路段流量，受到速度的影响
# def get_coefficent_matrix(adj):
#     """
#     We construct the linear equation system because the number of unknowns (edges)
#     is larger than the number of equations
#     (nodes) and is therefore a hyperstable problem, solved by pseudoinversion
#     :param adj: adjancy matrix,
#     :return:(num_nodes,num_edges)
#     """
#     num_edges = len(adj[np.where(adj==1)])
#     zeors_ = np.zeros((num_edges))
#     count = 0
#     a = []
#     for i in range(adj.shape[0]):#each row
#         zeros_0 = zeors_.copy()
#         for j in range(adj.shape[1]):#each cloum
#             if adj[i,j]!=0:
#                 zeros_0[count]=1
#                 count+=1
#         a.append(zeros_0)
#     return toarray(a)
#
# def solove_edges_flows(a,b):
#     """
#
#     :param a:(num_nodes,num_edges)->(functions,x)
#     :param b:(num_nodes,1)
#     :return:(num_edges,1)
#     """
#     return np.linalg.pinv(a).dot(b)
#
# def inverse_flow_matrix(adj,x):
#     """
#     Assignment of Section Flow to Adjacent Matrix Solved
#     :param adj:(N,N)
#     :param x:the solve from solove_edges_flows(a,b)
#     :return:adj matrix
#     """
#     adj[np.where(adj==1)]=np.squeeze(x)
#     return adj
