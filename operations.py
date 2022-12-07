# @Time    : 2022/11/20 16:10
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : operations
# @Project Name :STGMT-Tensorflow-implementation-master

import tensorflow as tf
import numpy as np


def get_shape(tensor):
    '''
    obtain the shape of tensor
    Args:
        tensor: a tensor (b,t,n,d)

    Returns:[b,t,n,d]

    '''
    return tensor.get_shape().as_list()

def positional_embedding(len_t,d_model):
    '''
    positional embedding for temporal ignore spatial
    Args:
        len_t: the length of temporal
        d_model: the last dimension

    Returns: a tensor (1,t,1,d)

    '''
    PE = np.array([[pos / np.power(10000, (i-i%2)/d_model) for i in range(d_model)] for pos in range(len_t)])
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])
    PE = PE* d_model ** 0.5
    PE = PE[np.newaxis, :, np.newaxis, :]
    outputs = tf.cast(PE, dtype=tf.float32)
    return outputs

def future_mask(att):
    '''
    Future masking in the first decoder of the time dimension to prevent future information leakage
    Args:
        att:(bnh,t_q,t_k)

    Returns:(bnh,t_q,t_k) after future_mask only att value or -inf

    '''
    padding_num = -2**32+1
    z,t_q,t_k = get_shape(att)
    diag_ = tf.ones_like(att[0,:,:])
    tril = tf.linalg.band_part(diag_,-1,0)
    future_masks = tf.tile(tf.expand_dims(tril,0),[z,1,1])
    paddings = tf.ones_like(future_masks)*padding_num
    return tf.where(tf.equal(future_masks,0),paddings,att)

def spatial_mask(tm):
    '''
    The point mask for points in the transfer matrix that are 0 is
    Args:
        tm:(n,n)

    Returns:(1,n,n) only 0 or -inf

    '''
    padding_num = -2 ** 32 + 1
    tm_mask = tf.zeros_like(tm)
    padding = tf.ones_like(tm_mask)*padding_num
    tm_mask = tf.where(tf.equal(tm,0),padding,tm_mask)
    return tf.expand_dims(tm_mask,0)

def meta_guide(inp,meta_ste):
    '''
    STE Guided learning to address spatio-temporal heterogeneity
    Args:
        inp:b,t,n,d,
        meta_ste:b,t,n,d,d

    Returns:

    '''
    inp_ = tf.expand_dims(inp,-1)#->(b,t,n,d,1)
    return tf.squeeze(tf.einsum('btndd,btndi->btndi',meta_ste,inp_),axis=-1)

def scale_dot_product_attention(q,k,v,tm,if_future,if_spatial):
    '''
    SDPA for multi head attention
    Args:
        q: (-1,q,d/h)
        k: (-1,k,d/h)
        v: (-1,k,d/h)
        tm: (n,n)
        if_future:boolen
        if_spatial:boolen

    Returns: (-1,q,d/h),(bnh,tq,tk)

    '''
    mat_qk = tf.matmul(q,tf.transpose(k,[0,2,1]))
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = mat_qk / tf.math.sqrt(dk)#-->(bnh,tq,tk)
    if if_future and tm is None:
        att = future_mask(scaled_attention_logits)
        att = tf.nn.softmax(att)
    elif if_spatial:
        tm_ = tf.expand_dims(tm,0)
        tm_mask = spatial_mask(tm)
        att = tf.add(scaled_attention_logits,tm_mask)
        att = tf.nn.softmax(att)*tm_
    else:
        att = scaled_attention_logits#-->(-1,q,k)
        att = tf.nn.softmax(att)
        # -->(-1,q,d/h)
    return tf.matmul(att,v),att
