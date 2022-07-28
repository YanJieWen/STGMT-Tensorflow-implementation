# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : layers
# @Project Name :metacode
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from Hyperparameters import Hyperparameters as hp

def projection(x):
    """

    :param x: inputs->(?,T,N,C)
    :return: (?,T,N,d)
    """
    return tf.layers.dense(x,hp.num_units,activation=tf.nn.leaky_relu)

def meta_guide_transform(qkv,weights):
    """

    :param qkv:(BN,T,d,1)
    :param weights:(BN,T,d,d)
    :return:(BN,T,d)
    """
    out = tf.squeeze(tf.matmul(weights,qkv),[-1])
    return out


def positional_embedding(inputs,zero_pad=True,scaled=True):#inputs_shape(BN(T),T(N),D)
    """
    Positional embedding layers
    :param inputs: inputs_shape(BN(T),T(N),D)
    :param zero_pad: a boolen
    :param scaled:a boolen
    :return:(?,T,d)
    """
    B = tf.shape(inputs)[0]
    batch_size, T_N ,num_units = inputs.get_shape().as_list()
    with tf.variable_scope('positinal_embedding',reuse=tf.AUTO_REUSE):
        position_ind = tf.tile(tf.expand_dims(tf.range(T_N), 0), [B,1])
        PE = np.array([
            [pos / np.power(10000, (i-i%2)/num_units) for i in range(num_units)]
            for pos in range(T_N)])
        PE[:, 0::2] = np.sin(PE[:, 0::2])
        PE[:, 1::2] = np.cos(PE[:, 1::2])
        lookup_table = tf.convert_to_tensor(PE,dtype=tf.float32)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)
        if scaled:
            outputs = outputs * num_units ** 0.5
        outputs = tf.cast(outputs,'float32')
        return outputs

def temproal_embedding_layers(x):
    """
    Time2Vec methods for time embedding
    :param x: (B,T,N,C)->(batch_size,time_seq,num_nodes,c)
    :return: (B,T,N,d)->(batch_size,time_seq,num_nodes,d)
    """
    with tf.variable_scope('Temporal_embedding', reuse=tf.AUTO_REUSE):
        B,T,N,C = x.get_shape().as_list()
        x_te = tf.reshape(tf.transpose(x, [0, 2, 1, 3]), [-1, T, C])
        origin = tf.layers.dense(x_te, 1, use_bias=True)
        sin_trans = tf.sin(tf.layers.dense(x_te, 64, use_bias=True))
        te_out = tf.concat((sin_trans, origin), axis=-1)
        te_out += positional_embedding(te_out)
        x_tes = tf.layers.dense(te_out, hp.num_units, use_bias=False)
        x_tes = tf.transpose(tf.reshape(x_tes, [-1, N, T, hp.num_units]), [0, 2, 1, 3])
        return x_tes

def spatio_embedding_layers(x):
    """
    Node2vec methods has benn adopted in the data deal processing for adjacency matrix
    :param x: a node embedding (N,C)
    :return: (N,d)
    """
    with tf.variable_scope('Spatio_embedding',reuse=tf.AUTO_REUSE):
        x_statics = tf.layers.dense(x,hp.num_units,use_bias=False)
    return x_statics

def Spatiotemporal_embeeding_layers(x_tes,x_statics,if_te=hp.if_te,if_se=hp.if_se):
    """

    :param x_tes: (B,T,N,d)
    :param x_statics: (N,d)
    :param if_te: if Ture the temporal_embedding is used
    :param if_se: if Ture the spatioal_embedding is used
    :return: (B,T,N,d)
    """
    with tf.variable_scope('SpatialTempralEmbedding', reuse=tf.AUTO_REUSE):
        B = tf.shape(x_tes)[0]
        T = x_tes.get_shape().as_list()[1]
        x_statics = tf.tile(tf.expand_dims(tf.expand_dims(x_statics, 0), 0), [B, T, 1, 1])
        if if_se and not if_te:
            return x_statics
        elif if_te and not if_se:
            return x_tes
        else:
            # ste_s = tf.concat((x_tes, x_statics), -1)
            ste_s = x_tes+x_statics
            ste_s = tf.layers.dense(ste_s, hp.num_units, use_bias=False)
            return ste_s
def meta_learner(ste_s,scope,num_weight_matrix):
    """

    :param ste_s: (B,T,N,d)
    :param num_weight_matrix: WQ,WV,WK guiad
    :return: a  list [?,T,N,1,dk,d]
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        B, T, N, d = ste_s.get_shape().as_list()
        meta_ = tf.layers.dense(ste_s, hp.num_units//2, use_bias=True, activation=tf.nn.leaky_relu)
        meta_ = tf.layers.dense(meta_, num_weight_matrix * hp.num_units * hp.num_units, use_bias=True)#placed most memory
        meta_ = tf.reshape(meta_, [-1, T, N, num_weight_matrix, hp.num_units, hp.num_units])  # ->(?,T,N,num_weight_m,dk,d)
        W_list = tf.split(meta_, num_weight_matrix, axis=3)
    return W_list

def temproal_attention(queries,keys,ste_q,ste_k,key_mask,drop_rate,scope,
                          if_training=True,if_keymask=True,if_casual=False,if_ste=True,if_meta=True):
    """

    :param queries: (B,T_q,N,d)
    :param keys: (B,T_k,N,d)
    :param ste_q: (B,T_q,N,d)
    :param ste_k: (B,T_k,N,d)
    :param key_mask: (B,T_q,1)
    :param drop_rate:a value
    :param if_training:a boolen
    :param if_keymask:a boolen
    :param if_casual:a boolen
    :param if_ste:a boolen
    :param if_meta:a boolen
    :return:(B,T_q,,N,d)
    """
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        B, T, N, d = queries.get_shape().as_list()
        B_, T_, N_, d_ = keys.get_shape().as_list()
        queries = tf.reshape(tf.transpose(queries,[0, 2, 1, 3]), [-1, T, d])#(BN,T,d)
        keys = tf.reshape(tf.transpose(keys,[0, 2, 1, 3]), [-1, T_, d_])#(BN,T,d)
        queries_ = tf.expand_dims(queries, -1)#(BN,T,d,1) for meta
        keys_ = tf.expand_dims(keys,-1)#(BN,T,d,1) for meta
        if not if_ste:
            Q = tf.layers.dense(queries,hp.num_units,use_bias=True)
            K = tf.layers.dense(keys,hp.num_units,use_bias=True)
            V = tf.layers.dense(keys,hp.num_units,use_bias=True)
            print('you are using nothing trick in your network!')
        else:
            if if_meta:
                guide_metrixes_q = meta_learner(ste_q,scope='queries',num_weight_matrix=1)#Wq,wk,wv not the same when interactivate
                guide_metrixes_kv = meta_learner(ste_k,scope='keys_values',num_weight_matrix=2)
                weight_matrixs_q = list(
                    map(lambda tensor: tf.reshape(tf.transpose(tensor, [0, 2, 1, 3, 4, 5]),
                    [-1, T, hp.num_units, hp.num_units]),guide_metrixes_q))#->[(BN,T,d,d),(BN,T,d,d),(BN,T,d,d)]
                weight_matrixs_kv  = list(
                    map(lambda tensor: tf.reshape(tf.transpose(tensor, [0, 2, 1, 3, 4, 5]),
                      [-1, T_, hp.num_units, hp.num_units]),guide_metrixes_kv))  # ->[(BN,T,d,d),(BN,T,d,d),(BN,T,d,d)]
                weight_matrixs_q.extend(weight_matrixs_kv)
                W_Q,W_K,W_V = weight_matrixs_q
                Q = meta_guide_transform(queries_,W_Q)#->(BN,T,d)
                K = meta_guide_transform(keys_,W_K)#->(BN,T,d)
                V = meta_guide_transform(keys_,W_V)#->(BN,T,d)
                print('you are using the meta in your network!')
            else:
                ste_q = tf.reshape(tf.transpose(ste_q,[0,2,1,3]),[-1,T,d])
                ste_k = tf.reshape(tf.transpose(ste_k, [0,2,1,3]), [-1,T_,d])
                Q = tf.layers.dense(queries+ste_q, hp.num_units, use_bias=True)
                K = tf.layers.dense(keys+ste_k, hp.num_units, use_bias=True)
                V = tf.layers.dense(keys+ste_k, hp.num_units, use_bias=True)
                print('you are using the STEs only "+" in your network!')
        #splite&concat
        Q_= tf.concat(tf.split(Q, hp.num_heads, axis=2), axis=0) # (BN*h,T_q,d/h)
        K_= tf.concat(tf.split(K, hp.num_heads, axis=2), axis=0) # (BN*h,T_k,d/h)
        V_ = tf.concat(tf.split(V, hp.num_heads, axis=2), axis=0) # (BN*h,T_k,d/h)
        #SDPA
        with tf.variable_scope('scaled_dot_product_attention',reuse=tf.AUTO_REUSE):
            d_k =  Q_.get_shape().as_list()[-1]
            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (BNh, T_q, T_k)
            outputs /= d_k ** 0.5
            #masking method
            padding_num = -2 ** 32 + 1  # an inf
            if if_keymask:  # padding masking
                key_masks = tf.to_float(key_mask)  # (BN, T_k,1)
                key_masks = tf.transpose(key_masks, [0, 2, 1])  # (BN, 1,T_k)
                key_masks = tf.tile(key_masks, [hp.num_heads, T, 1])  # (BNh, T_q, T_k)
                paddings = tf.ones_like(outputs) * padding_num
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (BNh, T_q, T_k)
            elif if_casual:
                diag_vals = tf.ones_like(outputs[0, :, :])  # generate a matrix->(T_q, T_k),filled 1
                tril = tf.linalg.band_part(diag_vals, -1, 0)#Upper triangular matrix
                future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                paddings = tf.ones_like(future_masks) * padding_num
                outputs = tf.where(tf.equal(future_masks, 0), paddings, outputs)
            #softmax
            outputs = tf.nn.softmax(outputs)
            attention = tf.transpose(outputs, [0, 2, 1])
            tf.summary.image("attention", tf.expand_dims(attention[:1], -1))  # save the first feature image
            #dropout
            outputs = tf.layers.dropout(outputs, rate=drop_rate, training=if_training)
            # weight_sum
            outputs = tf.matmul(outputs, V_)# (BN*h,T_q,d/h)
            outputs = tf.concat(tf.split(outputs, hp.num_heads, axis=0), axis=2)#(BN,T_q,d)
            # Residual connection
            outputs += queries  # ->#(BN,T_q,d)
            # Normalize
            outputs = ln(outputs)
            return tf.transpose(tf.reshape(outputs,[-1,N,T,d]),[0,2,1,3])

def spatio_attention(quries,keys,ste,transition_matrix,drop_rate,if_training=True,if_ste=True,if_meta=True,if_tmm=True):
    """

    :param quries: (B,T_q,N,d)
    :param keys: (B,T_q,N,d),the same
    :param ste: (B,T_q,N,d), the same
    :param transition_matrix: (num_matrix,N,N)
    :param drop_rate: a value
    :param if_training: a boolen
    :param if_ste: a boolen
    :param if_meta: a boolen
    :param if_tmm: a boolen
    :return: (B,T_q,N,d)
    """
    with tf.variable_scope('SpatioAtt', reuse=tf.AUTO_REUSE):
        B, T, N, D = quries.get_shape().as_list()
        B = tf.shape(quries)[0]
        quries = tf.reshape(quries,[-1,N,D])#(BT,N,D)
        keys= tf.reshape(keys, [-1, N, D])  # (BT,N,D)
        queries_ = tf.expand_dims(quries, -1)#(BT,N,d,1) for meta
        keys_ = tf.expand_dims(keys, -1)  # (BT,N,d,1) for meta
        number_tranm = transition_matrix.get_shape().as_list()[0]
        out = []
        for num in range(number_tranm):
            with tf.variable_scope('Transition_matrix{}'.format(num), reuse=tf.AUTO_REUSE):
                if if_ste:
                    if if_meta:
                        guide_matrix = meta_learner(ste,scope='qkv',num_weight_matrix=3)#[?,T,N,1,dk,d]
                        weight_matrixs = list(map(lambda tensor: tf.reshape(tensor,[-1, N, hp.num_units, hp.num_units]),
                                guide_matrix))  # ->[(BT,N,d,d),(BT,N,d,d),(BT,N,d,d)]
                        W_Q, W_K, W_V = weight_matrixs
                        Q = meta_guide_transform(queries_, W_Q)  # ->(BT,N,d)
                        K = meta_guide_transform(keys_, W_K)  # ->(BT,N,d)
                        V = meta_guide_transform(keys_, W_V)  # ->(BT,N,d)
                    else:
                        ste = tf.reshape(ste,[-1,N,D])
                        Q = tf.layers.dense(quries + ste, hp.num_units, use_bias=True)#->(BT,N,d)
                        K = tf.layers.dense(keys + ste, hp.num_units, use_bias=True)
                        V = tf.layers.dense(keys + ste, hp.num_units, use_bias=True)
                else:
                    Q = tf.layers.dense(quries, hp.num_units, use_bias=True)
                    K = tf.layers.dense(keys, hp.num_units, use_bias=True)
                    V = tf.layers.dense(keys, hp.num_units, use_bias=True)
                # splite&concat
                Q_ = tf.concat(tf.split(Q, hp.num_heads, axis=2), axis=0)  # (BT*h,N,d/h)
                K_ = tf.concat(tf.split(K, hp.num_heads, axis=2), axis=0)  # (BT*h,N,d/h)
                V_ = tf.concat(tf.split(V, hp.num_heads, axis=2), axis=0)  # (BT*h,N,d/h)
                with tf.variable_scope('scaled_dot_product_attention', reuse=tf.AUTO_REUSE):
                    d_k = Q_.get_shape().as_list()[-1]
                    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (BTh, N, N)
                    outputs /= d_k ** 0.5
                    padding_num = -2 ** 32 + 1  # an inf
                    #transition matrixes mask
                    if if_tmm:
                        tm_mask =tf.zeros_like(transition_matrix[num])#(N,N)
                        paddings = tf.ones_like(transition_matrix[num]) * padding_num
                        tm_mask = tf.where(tf.equal(transition_matrix[num],0),paddings,tm_mask)
                        tm_mask = tf.expand_dims(tm_mask,0)#(1,N,N)
                        # with tf.device('/cpu:0'):
                        #     tm_mask = tf.tile(tf.expand_dims(tm_mask,0),[B*T*hp.num_units,1,1])#placed most memory
                    #softmax
                    tmm = tf.expand_dims(transition_matrix[num],0)#brodcast
                    outputs = tf.add(outputs,tm_mask)#brodcast
                    outputs = tf.nn.softmax(outputs)*tmm
                    #store attention
                    attention = tf.transpose(outputs, [0, 2, 1])
                    tf.summary.image("attention", tf.expand_dims(attention[:1], -1))  # save the first feature image
                    # dropout
                    outputs = tf.layers.dropout(outputs, rate=drop_rate, training=if_training)
                    outputs = tf.matmul(outputs, V_)  # (BT*h,N,d/h)
                    outputs = tf.concat(tf.split(outputs, hp.num_heads, axis=0), axis=2)  # (BT,N,d)
            out.append(outputs)
        out = tf.reshape(tf.transpose(tf.convert_to_tensor(out),[1,2,3,0]),[-1,N,D*number_tranm])#(BT,N,D*NUM)
        out = tf.layers.dense(out,hp.num_units)#(BT,N,D)
        out = tf.layers.dropout(out, rate=drop_rate, training=if_training)
        out += quries
        out = ln(out)#(BT,N,D)
        return tf.reshape(out,[-1,T,N,D])


def feed_forward(inputs):
    """

    :param inputs: (B,T,N,d)
    :return: (B,T,N,,d)
    """
    with tf.variable_scope('FFN_layers',reuse=tf.AUTO_REUSE):
        outputs = tf.layers.dense(inputs,hp.num_units//4,activation=tf.nn.leaky_relu)
        outputs = tf.layers.dense(outputs,hp.num_units)
        outputs+=inputs
        outputs = ln(outputs)
    return outputs


def ln(inputs, epsilon=1e-8):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope('ln', reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer(), )
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

