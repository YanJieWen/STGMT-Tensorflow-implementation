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
            if i!=j:
                set_1 = datas[:24 * 7, i, :]#a week data, more data in not needed
                set_2 = datas[:24 * 7, j, :]
                # time_sim = np.sum(set_1 * set_2) / (np.linalg.norm(set_1) * np.linalg.norm(set_2))
                time_sim = np.exp(-(np.linalg.norm(set_1-set_2)/min(np.linalg.norm(set_1),
                np.linalg.norm(set_2))))
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
    for i in range(cal_matrix.shape[0]):
        for j in range(cal_matrix.shape[1]):
            if i==j:
                cal_matrix[i,j]=0
    diag = np.identity(cal_matrix.shape[0])
    cal_matrix+=diag#add self-loop
    x_norm = cal_matrix/np.sum(cal_matrix,axis=1,keepdims=True)
    return x_norm

def split_train_val_test(data):
    """
    shuffle the data sets
    :param data: (s,t,n,d)
    :return: train,val,test
    """
    np.random.seed(42)
    shuffle_id = np.random.permutation(len(data))
    train_datas_id = shuffle_id[:int(len(data)*0.6)]
    val_datas_id = shuffle_id[int(len(data)*0.6):(int(len(data)*0.2)+int(len(data)*0.6))]
    test_datas_id = shuffle_id[int(len(data)*0.2)+int(len(data)*0.6):]
    return data[train_datas_id],data[val_datas_id],data[test_datas_id]

def read_pkl(data_path):
    with open(data_path,'rb') as f:
        return pickle.load(f)

#cal metrcis->(bn,t)->(t,)
def cal_mae(gt,pred):
    mae = np.abs(gt-pred)
    return np.mean(mae,axis=0)
def cal_rmse(gt,pred):
    rmse = np.sqrt(np.mean(np.square(gt-pred),axis=0))
    return rmse
def cal_mape(gt,pred):
    multi_time_step = []
    for t in range(gt.shape[1]):
        gt_t = gt[:,t]
        pred_t = pred[:,t]
        gt_t_se = gt_t[np.where(gt_t>=10)]
        pred_t_se = pred_t[np.where(gt_t>=10)]
        mape = np.mean(np.abs(gt_t_se-pred_t_se)/np.abs(gt_t_se))
        multi_time_step.append(mape)
    return multi_time_step
