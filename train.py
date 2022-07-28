# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : train
# @Project Name :metacode

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from Hyperparameters import Hyperparameters as hp
from utils_ import *
from frameworks import *
from gen_batch import *
import time 
import random
# #Define GPU
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True


#load all datasets needed in framework
all_datas = read_pkl(hp.pkl_08)
train_data = all_datas[0]#gen by yield
val_data = all_datas[1]
zones = val_data.shape[2]
tms = toarray(all_datas[3])
num_sensors = train_data.shape[2]
if hp.if_onlydist:
    sim_matrix = toarray([tms[0]])#keep_dims
elif hp.if_onlytimesim:
    sim_matrix = toarray([tms[1]])
else:
    sim_matrix = tms
node2vec = all_datas[4]
scalar = all_datas[5]
predictions = np.zeros_like(val_data[:,hp.input_len:(hp.input_len+hp.output_len),:,:])
#load end
#Define the input datasets
x = tf.placeholder(dtype=tf.float32, shape=[None, hp.input_len, num_sensors, hp.out_units])  # ->(B,T,N,C)
y = tf.placeholder(dtype=tf.float32, shape=[None, hp.output_len, num_sensors, hp.out_units])  # ->(B,T2,N,C)
nodev = tf.placeholder(dtype=tf.float32, shape=[num_sensors, hp.d_node])  # ->(N,dc)
transition_matrices = tf.placeholder(dtype=tf.float32, shape=[sim_matrix.shape[0], num_sensors, num_sensors])#need to point a dims 
#build model
meta_transformer =Meta_transformer(x,y,nodev,transition_matrices)
enc_memory,src_mask,enc_stes = meta_transformer.encoder(if_training=True)
y_hat = meta_transformer.decoder(enc_memory,src_mask,enc_stes,if_training=True)
if hp.if_val:
    enc_memory_, src_mask_, enc_stes_ = meta_transformer.encoder(if_training=False)
    val_pred = meta_transformer.decoder(enc_memory_, src_mask_, enc_stes_,if_training=False)
#Define loss
loss = tf.losses.mean_squared_error(y,y_hat)#genearl datasets
# loss = tf.nn.l2_loss(y-y_hat)
# loss = tf.reduce_mean(tf.abs(y-y_hat))#senstive datsets
global_step = tf.train.get_or_create_global_step()#Create a global variable to record the number of steps
lr = tf.train.exponential_decay(hp.lr, global_step ,
decay_steps=5 * (len(train_data)//hp.batch_size), decay_rate=0.7, staircase=True)
optimizer = tf.train.AdamOptimizer(lr)
train_op = optimizer.minimize(loss, global_step=global_step)
tf.summary.scalar('lr', lr)
tf.summary.scalar("loss", loss)
tf.summary.scalar("global_step", global_step)
summaries = tf.summary.merge_all()
#start session
saver = tf.train.Saver(max_to_keep=3)
# with tf.Session(config=config) as sess:
with tf.Session() as sess:
    writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    sess.run(tf.global_variables_initializer())
    # saver.restore(sess,tf.train.latest_checkpoint('ckpt')) #future training
    init_error = hp.valid_thresh
    for i in range(hp.num_epochs):
        for j, inputs_ in enumerate(gen_batch(train_data,hp.batch_size)):
            start_time = time.time()
            _, _gs, summary, loss_ = sess.run([train_op, global_step, summaries, loss],
                                              feed_dict={x: inputs_[:, :hp.input_len,:,:],
                                                         y: inputs_[:, hp.input_len:(hp.input_len + hp.output_len),:,:],
                                                         nodev:node2vec,
                                                         transition_matrices:sim_matrix})
            writer.add_summary(summary, _gs)
            # autoregression for each step
            if j%20==0:
        # autoregression for each step
                if hp.if_val:  # set value is False to avoid memoryout
                    len_data = len(val_data)
                    idx = np.arange(len_data)
                    np.random.shuffle(idx)
                    id = random.choice(idx)
                    start_idx = id
                    end_idx = id+hp.batch_size
                    if end_idx>len_data:
                        end_idx = len_data
                    slc = idx[start_idx:end_idx]
                    for step in range(hp.output_len):
                        _predictions = sess.run(val_pred, feed_dict={x: val_data[slc,:hp.input_len,:,:], y: predictions[slc],
                                                                    nodev:node2vec,
                                                                    transition_matrices:sim_matrix})
                        predictions[slc,step,:,:] = _predictions[:, step,:,:]
                    predictions_0 = inverse_minmaxsca(predictions[slc,:,:,:].reshape(-1,zones,hp.out_units),scalar)
                    gt_0 = inverse_minmaxsca(val_data[slc,hp.input_len:,:,:].reshape(-1,zones,hp.out_units),scalar)#the first time point
                    error = np.mean(np.abs(predictions_0-gt_0))
                    print('The validation MAE value is {:.2f}, for the {} steps in {} epochs.'.format(error,j,i))
                    if error<=init_error:
                        init_error = error
                        saver.save(sess=sess, save_path=hp.ckpt_path, global_step=(i + 1))
                else:
                    if i % 10 == 0 and j % (20 * 3) == 0 or i == hp.num_epochs - 1:
                        saver.save(sess=sess, save_path=hp.ckpt_path, global_step=(i + 1))
                print('The current loss value is {:.5f}, for the {} steps in {} epochs.Per step use {:.2f} s'.format(loss_, j, i,time.time()-start_time))
    sess.close()


