# @Time    : 2022/6/9 11:23
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : data_fac
# @Project Name :metacode

from Hyperparameters import Hyperparameters as hp
import numpy as np
import pickle
import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import glob
import os
from utils_ import *

def get_data_path(dir,type):
    """

    :param dir: rootdir path like './data/PEMS04'
    :param type: .npz,.csv
    :return: a full path
    """
    files = []
    for name in glob.glob('{}/*.{}'.format(dir,type)):
        files.append(name)
    return files[0]

def read_flow_data(dir):
    """

    :param dir: rootdir path like './data/PEMS04'
    :return: (points,sensors,flow)
    """
    pe = np.load(get_data_path(dir,'npz'))
    datas = pe['data'][:,:,0,None]#for pkl
    # datas = np.load(get_data_path(dir,'npy'))#for NYC
    return datas

def split_data_follow_day(datas,train_days,val_days):
    """

    :param datas: ((points,sensors,flow))
    :param train_days: value
    :param val_days: value
    :return: train,val,test data
    """
    train_points = train_days*288#need to change 
    val_points = (train_days+val_days)*288#need to change 
    return [datas[:train_points,:,:],datas[train_points:val_points,:,:],datas[val_points:]]

def get_datasets(datas):
    """

    :param datas: train or val or test
    :return: train or val or test with shape(num_points,sequence_len,sensors,flow)
    """
    datasets = []
    for i in range(len(datas) - hp.input_len - hp.output_len):
        datasets .append(datas[i:i + hp.input_len + hp.output_len])
    data = toarray(datasets)
    return data
def read_graph(dir):
    """

    :param dir: a dirpath
    :return: dataframe contain statics graph
    """
    df = pd.read_csv(get_data_path(dir,'csv'))
    return df
def build_adjmatrix(df):
    """

    :param df: dataframe
    :return: (N,N) adj matrix with the distance of the each node
    """
    nodes_id = list(set(list(set(df['from'].values)) + list(set(df['to'].values))))
    nodes_id=sorted(nodes_id)
    nodes_id = np.array(nodes_id)
    adj_init = np.zeros((len(nodes_id), len(nodes_id)))
    for edge_idx in range(len(df)):
        from_id = df.iloc[edge_idx]['from']
        to_id = df.iloc[edge_idx]['to']
        adj_init[np.where(nodes_id==from_id), np.where(nodes_id==to_id)] = df.iloc[edge_idx]['cost']
    adj_matrix = adj_init
    return adj_matrix


def data_preprocessing(dir,train_day,val_day):
    """

    :param dir: like './data/PEMS04
    :param train_day: a value
    :param val_day: a value
    :return: [[train],[val],[test],[dism,tsimm],node2v,sca]
    """
    #get train,val,test
    # dir = './data/PEMS04'
    datas =read_flow_data(dir)
    datas_,sca = minmaxsca(datas)#sca is needed
    datasets = split_data_follow_day(datas_,train_day,val_day)#a list
    #dynamic datasets results
    all_datas = list(map(get_datasets,datasets))#the xs and ys,[[train],[val],[test]]
    #to get statics graph
    graph_df = read_graph(dir)
    adj_matrix = build_adjmatrix(graph_df)
    dis_sim = sim_distance(adj_matrix, graph_df)
    dis_norm = norm_adjmatrix(dis_sim)
    time_sim_norm = time_sim_matrix(datas)
    #to get node2vec
    graph = build_graph(get_data_path(dir,'csv'))
    node2vec = Node2Vec(graph, dimensions=hp.d_node, walk_length=hp.walk_len, num_walks=hp.num_walks,
                        p=hp.p, q=hp.q, workers=hp.workers)
    model = node2vec.fit()
    node_vec = toarray([model.wv[str(node)] for node in sorted(graph.nodes())])  # (N,dc)，if error you coudle model.wv due to the version of gensim
    #all data summary
    all_datas.append([dis_norm,time_sim_norm])
    all_datas.append(node_vec)
    all_datas.append(sca)
    return all_datas

def to_pkl(datas,pkl_path):
    with open(pkl_path,'wb') as f:
        pickle.dump(datas,f)


def main():
    all_datas = data_preprocessing(hp.data_08,44,5)#PEMS08->44,5;NYC->168,8;PEMS04->44,7(59)
    to_pkl(all_datas,hp.pkl_08)

if __name__ == '__main__':
    main()


