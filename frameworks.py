# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : frameworks
# @Project Name :metacode

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from layers import *
from Hyperparameters import Hyperparameters as hp

class Meta_transformer():
    def __init__(self,x_tensor,y_tensor,node2vec,transition_matrixs):
        self.xs = x_tensor#（B,T,N,1）
        self.ys = y_tensor#（B,T,N,1）
        self.node2vec = node2vec#(N,d//2)
        self.transition_matrixs = transition_matrixs
    def encoder(self,if_training=True):
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            #STEs_embedding
            enc_tes = temproal_embedding_layers(self.xs)#（B,T,N,d）
            enc_statics = spatio_embedding_layers(self.node2vec)#(N,d)
            enc_stes = Spatiotemporal_embeeding_layers(enc_tes,enc_statics,hp.if_te,hp.if_se)#(B,T,N,d),need in decoder
            #key_mask
            B,T,N,d = self.xs.get_shape().as_list()
            xs = tf.reshape(tf.transpose(self.xs,[0, 2, 1, 3]), [-1, T, d])
            src_mask = tf.expand_dims(tf.sign(tf.reduce_sum(tf.abs(xs),axis=-1)),-1)#（BN,T_q,1）,need in decoder
            enc = projection(self.xs)
            # To sturcture a sandwiches Transformer
            #blcock_begin
            for i in range(hp.num_begin_blocks):
                with tf.variable_scope('num_blocks{}_begin'.format(i), reuse=tf.AUTO_REUSE):
                    enc = temproal_attention(queries=enc, keys=enc, ste_q=enc_stes,scope='SELF_ATT',
                                             ste_k=enc_stes, key_mask=src_mask, drop_rate=hp.drop_rate,
                                             if_training=if_training,
                                             if_keymask=False, if_casual=False, if_ste=hp.if_ste,
                                             if_meta=hp.if_meta)  # (B,T,N,d)
                    enc = spatio_attention(quries=enc, keys=enc, ste=enc_stes,
                                           transition_matrix=self.transition_matrixs,
                                           drop_rate=hp.drop_rate, if_training=if_training, if_ste=hp.if_ste,
                                           if_meta=hp.if_meta,
                                           if_tmm=True)  # (B,T,N,d)
            #blcock medium
            for i in range(hp.num_medium_blocks):
                with tf.variable_scope('num_blocks{}_midume'.format(i),reuse=tf.AUTO_REUSE):
                    enc = temproal_attention(queries=enc,keys=enc,ste_q=enc_stes,scope='SELF_ATT',
                                                ste_k=enc_stes,key_mask=src_mask,drop_rate=hp.drop_rate,if_training=if_training,
                                                if_keymask=False,if_casual=False,if_ste=hp.if_ste,if_meta=hp.if_meta)#(B,T,N,d)
                    enc = spatio_attention(quries=enc,keys=enc,ste=enc_stes,transition_matrix=self.transition_matrixs,
                                           drop_rate=hp.drop_rate,if_training=if_training,if_ste=hp.if_ste,if_meta=hp.if_meta,
                                           if_tmm=True)#(B,T,N,d)
                    enc = feed_forward(enc)#(B,T,N,d)
            #block end
            for i in range(hp.num_end_blocks):
                with tf.variable_scope('num_blocks{}_end'.format(i), reuse=tf.AUTO_REUSE):
                    enc = feed_forward(enc)#(B,T,N,d)
        enc_memory = enc
        return enc_memory,src_mask,enc_stes

    def decoder(self,memory,key_mask,enc_stes,if_training=True):
        with tf.variable_scope('Decoder',reuse=tf.AUTO_REUSE):
            #build a shifted right decoder_input for teaching forcing
            B,T,N,c = self.ys.get_shape().as_list()
            ys = tf.reshape(tf.transpose(self.ys,[0,2,1,3]),[-1,T,c])
            B_x,T_x,N_x,d_x = self.xs.get_shape().as_list()
            xs = tf.reshape(tf.transpose(self.xs,[0, 2, 1, 3]), [-1, T_x, d_x])
            decoder_inp = tf.transpose(tf.reshape(tf.concat((xs[:,-1:,:],ys[:,:-1,:]),1),[-1,N,T,c]),[0,2,1,3])
            # decoder_inp = tf.transpose(tf.reshape(tf.concat((tf.ones_like(ys[:,:1,:])*2,ys[:,:-1,:]),1),[-1,N,T,c]),[0,2,1,3])
            #ste
            dec_tes = temproal_embedding_layers(decoder_inp)
            dec_statics = spatio_embedding_layers(self.node2vec)
            dec_stes = Spatiotemporal_embeeding_layers(dec_tes,dec_statics,hp.if_te,hp.if_se)
            dec = projection(decoder_inp)
            #sandwiches transformer
            # blcock_begin
            for i in range(hp.num_begin_blocks):
                with tf.variable_scope('num_blocks{}_begin'.format(i), reuse=tf.AUTO_REUSE):
                    dec = temproal_attention(queries=dec, keys=dec, ste_q=dec_stes,scope='SELF_ATT',
                                             ste_k=dec_stes, key_mask=key_mask, drop_rate=hp.drop_rate,
                                             if_training=if_training,
                                             if_keymask=False, if_casual=True, if_ste=hp.if_ste,
                                             if_meta=hp.if_meta)  # (B,T,N,d)
                    dec= spatio_attention(quries=dec, keys=dec, ste=dec_stes,
                                           transition_matrix=self.transition_matrixs,
                                           drop_rate=hp.drop_rate, if_training=if_training, if_ste=hp.if_ste,
                                           if_meta=hp.if_meta,
                                           if_tmm=True)  # (B,T,N,d)
                    #intercative attention for temproal
                    dec = temproal_attention(queries=dec, keys=memory, ste_q=dec_stes,scope='INTER_ATT',
                                             ste_k=enc_stes, key_mask=key_mask, drop_rate=hp.drop_rate,
                                             if_training=if_training,
                                             if_keymask=False, if_casual=False, if_ste=hp.if_ste,
                                             if_meta=hp.if_meta)  # (B,T,N,d)
            # blcock medium
            for i in range(hp.num_medium_blocks):
                with tf.variable_scope('num_blocks{}_midume'.format(i), reuse=tf.AUTO_REUSE):
                    dec = temproal_attention(queries=dec, keys=dec, ste_q=dec_stes,scope='SELF_ATT',
                                             ste_k=dec_stes, key_mask=key_mask, drop_rate=hp.drop_rate,
                                             if_training=if_training,
                                             if_keymask=False, if_casual=True, if_ste=hp.if_ste,
                                             if_meta=hp.if_meta)  # (B,T,N,d)
                    dec = spatio_attention(quries=dec, keys=dec, ste=dec_stes,
                                           transition_matrix=self.transition_matrixs,
                                           drop_rate=hp.drop_rate, if_training=if_training, if_ste=hp.if_ste,
                                           if_meta=hp.if_meta,
                                           if_tmm=True)  # (B,T,N,d)
                    # intercative attention for temproal
                    dec = temproal_attention(queries=dec, keys=memory, ste_q=dec_stes,scope='INTER_ATT',
                                             ste_k=enc_stes, key_mask=key_mask, drop_rate=hp.drop_rate,
                                             if_training=if_training,
                                             if_keymask=False, if_casual=False, if_ste=hp.if_ste,
                                             if_meta=hp.if_meta)  # (B,T,N,d)
                    dec= feed_forward(dec)  # (B,T,N,d)
            # block end
            for i in range(hp.num_end_blocks):
                with tf.variable_scope('num_blocks{}_end'.format(i), reuse=tf.AUTO_REUSE):
                    dec = feed_forward(dec)  # (B,T,N,d)
        with tf.variable_scope('Prediction',reuse=tf.AUTO_REUSE):
            # weight_variable = tf.get_variable('out_weight',shape=[hp.num_units,hp.out_units],initializer=tf.random_normal_initializer(),dtype=tf.float32)
            # tf.add_to_collection(name='weight_decay', value=0.03*tf.nn.l2_loss(weight_variable))
            # y_hat = tf.einsum('btnd,dk->btnk',dec,weight_variable,name='y_hat')
            y_hat = tf.layers.dense(dec,hp.out_units,use_bias=False)#the same with the einsum
            # tf.summary.histogram('d1',weight_variable)
            # y_hat = tf.layers.dense(dec,hp.out_units,activation=tf.nn.sigmoid,name='y_hat')
        return y_hat
# #
# def main():
#     x = tf.placeholder(dtype=tf.float32, shape=[32, 4, 80, 2])  # ->(B,T,N,C)
#     y = tf.placeholder(dtype=tf.float32, shape=[32, 10, 80, 2])  # ->(B,T2,N,C)
#     nodev = tf.placeholder(dtype=tf.float32, shape=[80, 64])  # ->(N,dc)
#     transition_matrices = tf.placeholder(dtype=tf.float32, shape=[3, 80, 80])  # ->(n_m,N,N)
#     meta_former =Meta_transformer(x,y,nodev,transition_matrices)
#     enc_memory,src_mask,enc_stes = meta_former.encoder(if_training=True)
#     y_hat = meta_former.decoder(enc_memory,src_mask,enc_stes,if_training=True)
#     print(y_hat)

# if __name__ == '__main__':
#     main()


