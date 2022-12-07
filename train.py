# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : train
# @Project Name :metacode

import tensorflow as tf
import numpy as np
from Hyperparameters import Hyperparameters as hp
from utils_ import *
from frameworks import *
from operations import *
import time 
import random

#step1：Prepare dataset pipline
data = read_pkl(hp.pkl_08)
train= data[0]
val = data[1]
test = data[2]
tms = toarray(data[3])
node2vec = data[4]
scalar = data[5]
train_dataset = tf.data.Dataset.from_tensor_slices((train[:,:hp.input_len,:,:],train[:,-hp.input_len:,:,:]))
val_dataset = tf.data.Dataset.from_tensor_slices((val[:,:hp.input_len,:,:],val[:,-hp.input_len:,:,:]))
train_dataset = train_dataset.cache()#for train dataset
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(hp.batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=1000).batch(hp.batch_size)
val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
#step2：define loss object
loss_object = tf.keras.losses.MeanAbsoluteError(name='mean_absolute_error')
def loss_fn(pred,gt):
    return tf.reduce_mean(loss_object(gt,pred))
#step3:define metric object
# def rsqure(gt,pred):
#     fenzi = tf.reduce_sum(tf.math.square(pred-gt))
#     fenmu = tf.reduce_sum(tf.math.square(gt-tf.reduce_mean(gt)))
#     return tf.cast((1-fenzi/fenmu),tf.float32)
# class R_square(tf.keras.metrics.Metric):
#     def __init__(self):
#         super().__init__()
#         self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
#         self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())
#     def update_state(self,gt,pred):
#         values = rsqure(gt,pred)
#         self.total.assign_add(tf.reduce_sum(values))
#         self.count.assign_add(tf.cast(tf.size(gt)),tf.float32)
#     def result(self):
#         return self.total/self.count
train_loss = tf.keras.metrics.MeanSquaredError(name='train_loss')
#step4:Define optimizer
learning_rate = CustomSchedule(hp.num_units)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9,clipvalue=1.0)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
#step5:Define training and checkpoint
stgmt = STGMT(d_model=hp.num_units,num_heads=hp.num_heads)
checkpoint_path = hp.ckpt_path
ckpt = tf.train.Checkpoint(stgmt=stgmt,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# 如果检查点存在，则恢复最新的检查点。
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')#if future training
#step6:Define training step
# train_step_signature = [
#     tf.TensorSpec(shape=(None, None,None,None), dtype=tf.float32),
#     tf.TensorSpec(shape=(None, None,None,None), dtype=tf.float32),
# ]
@tf.function
def train_step(enc_inp,dec_inp,n2v, tms):
    with tf.GradientTape() as tape:
        final_output, enc_atts, dec_atts = stgmt(enc_inp,dec_inp,n2v,tms,if_te=True,if_se=True,if_ste=True,if_meta=True,training=True)
        loss = loss_fn(final_output,dec_inp)
    gradients = tape.gradient(loss, stgmt.trainable_variables)
    optimizer.apply_gradients(zip(gradients, stgmt.trainable_variables))
    train_loss(dec_inp,final_output)
    # train_accuarcy()
#eval function
def evaluate(enc_inp,n2v,tms):
    b,t,n,d = get_shape(enc_inp)
    dec_inp = tf.zeros((b,hp.output_len,n,d))
    for step in range(dec_inp.shape[1]):
        outs, _, _ = stgmt(enc_inp, dec_inp, n2v, tms, if_te=True, if_se=True, if_ste=True,
                                                 if_meta=True, training=False)
        outs_ = outs.numpy()
        dec_inp_ = dec_inp.numpy()
        dec_inp_[:, step, :, :] = outs_[:, step, :, :]
        dec_inp = tf.convert_to_tensor(dec_inp_,tf.float32)
        # dec_inp_list = tf.unstack(dec_inp)
        # outs_list = tf.unstack(outs)
        # dec_inp_list[:,step,:,:]=outs_list[:,step,:,:]
        # dec_inp = tf.stack(dec_inp_list)
    return dec_inp#->(b,t,n,d)


#train processing
init_val = hp.init_val
for epoch in range(hp.num_epochs):
    start = time.time()
    train_loss.reset_states()
    # train_accuarcy.update_state(gt,pred)
    for (batch,(enc_inp,dec_inp)) in enumerate(train_dataset):
        train_step(enc_inp,dec_inp,node2vec, tms)
        if batch%50==0:
            print('Epoch {} /Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, train_loss.result()))
        if (epoch+1)%5==0 and batch%50==0:
            if hp.if_val:#if use validation data
                count = 0
                errors = []
                while count<10:
                    enc_inp,gt = next(iter(val_dataset))
                    pred_ = evaluate(enc_inp,node2vec,tms)
                    pred = scalar.inverse_transform(tf.reshape(pred_,[-1,hp.out_units]))
                    gt = scalar.inverse_transform(tf.reshape(gt,[-1,hp.out_units]))
                    mae = tf.reduce_mean(tf.abs(pred-gt))
                    errors.append(mae)
                    count+=1
                mae = tf.reduce_mean(tf.convert_to_tensor(errors,tf.float32))
                if mae.numpy()<init_val:
                    ckpt_save_path = ckpt_manager.save()
                    print('Saving checkpoint for epoch {} at {}-->MAE is {:.4f}'.format(epoch + 1,
                                                                       ckpt_save_path,mae.numpy()))
                    init_val = mae.numpy()

            else:
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                    ckpt_save_path))
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))



