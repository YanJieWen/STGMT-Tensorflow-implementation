from Hyperparameters import Hyperparameters as hp
from utils_ import *
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from frameworks import *
#Another way to test datasets
num_sensors = 330
x = tf.placeholder(dtype=tf.float32, shape=[None, hp.input_len, num_sensors, hp.out_units])  # ->(B,T,N,C)
y = tf.placeholder(dtype=tf.float32, shape=[None, hp.output_len, num_sensors, hp.out_units])  # ->(B,T2,N,C)
nodev = tf.placeholder(dtype=tf.float32, shape=[num_sensors, hp.d_node])  # ->(N,dc)
transition_matrices = tf.placeholder(dtype=tf.float32, shape=[2, num_sensors, num_sensors])#need to point a dims 
#build model
meta_transformer =Meta_transformer(x,y,nodev,transition_matrices)
enc_memory_, src_mask_, enc_stes_ = meta_transformer.encoder(if_training=False)
predict = meta_transformer.decoder(enc_memory_, src_mask_, enc_stes_,if_training=False)
#get gt
datas = read_pkl(hp.NYC_)
test_data = datas[2][:2]#test data,(B,T_h+T_f,N,F),get 10batch
tms = np.array(datas[3])#(2,N,N)
node2vec = datas[4]#(N,64)
scalar = datas[5]
predictions = np.zeros_like(test_data[:,hp.input_len:,:,:])
test_sess = tf.Session()
saver = tf.train.Saver()
test_sess.run(tf.global_variables_initializer())
saver.restore(test_sess, tf.train.latest_checkpoint('ckpt'))
for step in range(hp.output_len):
    _predictions = test_sess.run(predict, feed_dict={x: test_data[:, :hp.input_len,:,:], y: predictions,nodev:node2vec,transition_matrices:tms})
    predictions[:, step,:,:] = _predictions[:, step,:,:]

pred = np.squeeze(scalar.inverse_transform(predictions.transpose([0,2,1,3]).reshape([-1,hp.out_units])).reshape([-1,hp.output_len,hp.out_units]))
gt = np.squeeze(scalar.inverse_transform(test_data[:,hp.input_len:,:,:].transpose([0,2,1,3]).reshape([-1,hp.out_units])).reshape([-1,hp.output_len,hp.out_units]))
print(np.mean(np.abs(pred-gt)))