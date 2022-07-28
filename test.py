# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : test
# @Project Name :metacode
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from Hyperparameters import Hyperparameters as hp
from utils_ import *
import pickle
import random
import pandas as pd
#get_data
datas = read_pkl(hp.pkl_08)
test_data = datas[2]#test data,(B,T_h+T_f,N,F)
tms = np.array(datas[3])#(2,N,N)
node2vec = datas[4]#(N,64)
scalar = datas[5]
zones = test_data.shape[2]
#restore model
test_sess = tf.Session()
saver = tf.train.import_meta_graph('./ckpt/weight-61.meta')
graph = tf.get_default_graph()
saver.restore(test_sess, tf.train.latest_checkpoint('ckpt'))
#define source data
len_test = len(test_data)
idx = np.arange(len_test)
decoder_output_ = []
output_datas_ = []
for i in range(50):
    np.random.shuffle(idx)
    id = random.choice(idx)
    end_idx = id+hp.batch_size
    if end_idx>=len_test:
        end_id = len_test
    slc = idx[id:end_idx]
    input_datas = test_data[slc,:hp.input_len,:,:]#only one points avoid memory out
    output_datas = test_data[slc,hp.input_len:,:,:]
    decoder_output = np.ones_like(output_datas) 
    #define placehodler
    x_input = graph.get_tensor_by_name('Placeholder:0')
    decoder_input = graph.get_tensor_by_name('Placeholder_1:0')
    nodev  = graph.get_tensor_by_name('Placeholder_2:0')
    tms_ = graph.get_tensor_by_name('Placeholder_3:0')
    # y_hat = graph.get_tensor_by_name('Prediction/y_hat/Einsum:0')
    y_hat = graph.get_tensor_by_name('Prediction/dense/Tensordot:0') 
    #autoregression
    for j in range(hp.output_len):
        _pred = test_sess.run(y_hat, feed_dict={x_input:input_datas,
        decoder_input:decoder_output,nodev:node2vec,tms_:tms})
        decoder_output[:,j,:,:] = _pred[:, j,:,:]#(B,T_f,N,1)
    if decoder_output.shape[0]==hp.batch_size:
        decoder_output_.append(decoder_output)#可能有不等于16个batch的情况，因此无法转为数组
        output_datas_.append(output_datas)
    else:
        continue
decoder_output = np.reshape(np.array(decoder_output_),[-1,hp.output_len,zones ,1])
output_datas = np.reshape(np.array(output_datas_),[-1,hp.output_len,zones ,1])

#get error
pred = np.squeeze(scalar.inverse_transform(decoder_output.transpose([0,2,1,3]).reshape([-1,hp.out_units])).reshape([-1,hp.output_len,hp.out_units]))
print(pred)
to_pred = np.reshape(pred,(-1,zones ,hp.output_len))
df_pred=  pd.DataFrame(to_pred[5,:,:])
df_pred.to_csv('./pred.csv')
#(BN,T)
gt = np.squeeze(scalar.inverse_transform(output_datas.transpose([0,2,1,3]).reshape([-1,hp.out_units])).reshape([-1,hp.output_len,hp.out_units]))
print(gt)
to_gt = np.reshape(gt,(-1,zones,hp.output_len))
df_gt=  pd.DataFrame(to_gt[5,:,:])
df_gt.to_csv('./gt.csv')
#(BN,T)
#get eval
MAE = np.mean(np.abs(pred-gt),axis=0)#（1，12）
multi_MAE = MAE[hp.step_index]#(1,4)
#RMSE
RMSE = np.sqrt(np.mean(np.square(pred-gt),axis=0))
multi_RMSE = RMSE[hp.step_index]
#MAPE
MAPE = np.mean(np.abs(pred-gt)/((np.abs(pred)+np.abs(gt))),axis=0)
multi_MAPE = MAPE [hp.step_index]
#to store multi step
multi_error = toarray([MAE,RMSE,MAPE])
df_e = pd.DataFrame(multi_error)
df_e.to_csv('./multi_error_our.csv')
print('The MAE,RMSE,MAPE in the 3,6,9,12 steps are {},{},{}'.format(multi_MAE,multi_RMSE,multi_MAPE))
print(np.mean(MAE),np.mean(RMSE),np.mean(MAPE))
# tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
# for tensor_name in tensor_name_list[:5000]:
#     print(tensor_name)