# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : test
# @Project Name :metacode

import tensorflow as tf
import numpy as np
from Hyperparameters import Hyperparameters as hp
from utils_ import *
import pickle
import random
import pandas as pd

from frameworks import *

#step1：get test data
data = read_pkl(hp.pkl_08)
test = data[2]
tms = toarray(data[3])
node2vec = data[4]
scalar = data[5]
test_dataset = tf.data.Dataset.from_tensor_slices((test[:,:12,:,:],test[:,-12:,:,:]))
test_dataset = test_dataset.shuffle(buffer_size=1000).batch(hp.batch_size)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#load model sturctre
stgmt = STGMT(d_model=hp.num_units,num_heads=hp.num_heads)
#load weights with ckpt form
checkpoint = tf.train.Checkpoint(stgmt=stgmt)
checkpoint.restore(tf.train.latest_checkpoint(hp.ckpt_path)).expect_partial()
print('The latest weight has been restored!')
def evaluate(enc_inp, n2v, tms,model):
  b, t, n, d = get_shape(enc_inp)
  dec_inp = tf.ones((b, hp.output_len, n, d))
  for step in range(dec_inp.shape[1]):
    outs, _, _ = model(enc_inp, dec_inp, n2v, tms, if_te=True, if_se=True, if_ste=True,
                       if_meta=True, training=False)
    outs_ = outs.numpy()
    dec_inp_ = dec_inp.numpy()
    dec_inp_[:, step, :, :] = outs_[:, step, :, :]
    dec_inp = tf.convert_to_tensor(dec_inp_, tf.float32)
  return dec_inp  # ->(b,t,n,d)
#eval
mae = []
rmse = []
mape = []
gts = []
preds = []
# errors = []
count=0
for batch,(enc_inp,gt) in enumerate(test_dataset):
  b,t,n,d = get_shape(gt)
  pred_ = evaluate(enc_inp,node2vec,tms,stgmt)
  pred = scalar.inverse_transform(tf.reshape(pred_,[-1,1]))#btn,d
  gt = scalar.inverse_transform(tf.reshape(gt,[-1,1]))
  gt0 = np.reshape(np.transpose(np.reshape(gt,(-1,hp.output_len,n,d)),[0,2,1,3]),(-1,hp.output_len))
  pred0 = np.reshape(np.transpose(np.reshape(pred,(-1,hp.output_len,n,d)),[0,2,1,3]),(-1,hp.output_len))
  preds.append(pred0)
  gts.append(gt0)
  mae.append(cal_mae(gt0,pred0)[np.newaxis,:])
  rmse.append(cal_rmse(gt0,pred0)[np.newaxis,:])
  mape.append(toarray(cal_mape(gt0,pred0))[np.newaxis,:])

mae = np.concatenate(mae,axis=0)
rmse = np.concatenate(rmse,axis=0)
mape = np.concatenate(mape,axis=0)
gts = np.concatenate(gts,axis=0)
preds = np.concatenate(preds,axis=0)
print(np.mean(mae,axis=0))
print('*'*30)
print(np.mean(rmse,axis=0))
print('*'*30)
print(np.mean(mape,axis=0))
print('*'*30)
errors = [np.mean(mae,axis=0),np.mean(rmse,axis=0),np.mean(mape,axis=0)]
df = pd.DataFrame(errors)
df.to_csv('./gap.csv')
writer = pd.ExcelWriter('ana.xlsx')
df_gt = pd.DataFrame(gts[:1000,:])
df_pred = pd.DataFrame(preds[:1000,:])
df_gt.to_excel(writer,'sheet1')
df_pred.to_excel(writer,'sheet2')
writer.save()
