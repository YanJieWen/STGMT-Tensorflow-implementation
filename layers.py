# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : layers
# @Project Name :metacode
import tensorflow as tf
import numpy as np
from Hyperparameters import Hyperparameters as hp
from operations import *

class Projection(tf.keras.layers.Layer):
    def __init__(self,d_model,**kwargs):
        '''
        Non-Linear transformation of historical input
        Args:
            d_model: transform dimension
            **kwargs: None
        '''
        super(Projection,self).__init__(**kwargs)
        self.d_model = d_model
        self.project = tf.keras.layers.Dense(self.d_model,activation=tf.keras.activations.relu,use_bias=False)
    def call(self,inp):
        return self.project(inp)

class Temporal_embedding_layer(tf.keras.layers.Layer):
    def __init__(self,d_model,t_len,**kwargs):
        '''
        Te with time2vec->(B,T,N,D)
        Args:
            d_model: a value
            t_len: a value
            **kwargs:
        '''
        super(Temporal_embedding_layer,self).__init__(**kwargs)
        self.d_model = d_model
        self.t_len = t_len
        self.dense1 = tf.keras.layers.Dense(1,activation=tf.keras.activations.linear,use_bias=True)
        self.dense2 = tf.keras.layers.Dense(self.d_model,activation=tf.keras.activations.linear,use_bias=True)
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dense3 = tf.keras.layers.Dense(self.d_model,activation=tf.keras.activations.linear,use_bias=False)
        self.dense4 = tf.keras.layers.Dense(self.d_model,activation=tf.keras.activations.linear,use_bias=False)
    def call(self,inp):
        B,T,N,C = get_shape(inp)
        inp_ = tf.reshape(tf.transpose(inp,perm=[0,2,1,3]),(-1,T,C))
        origin = self.dense1(inp_)
        sin_ = tf.math.sin(self.dense2(inp_))
        te_out = self.concat((sin_,origin))
        te_out = self.dense3(te_out)
        out_ = tf.transpose(tf.reshape(te_out, (-1, N, T, self.d_model)), perm=[0, 2, 1, 3])
        out_ += positional_embedding(self.t_len, self.d_model)
        out_ = self.dense4(out_)
        return out_

class Spatial_embedding_layers(tf.keras.layers.Layer):
    def __init__(self,d_model,**kwargs):
        '''
        node2vec after linear transform->(N,d)
        Args:
            d_model: a value
            **kwargs:
        '''
        super(Spatial_embedding_layers,self).__init__(**kwargs)
        self.d_model= d_model
        self.dense = tf.keras.layers.Dense(self.d_model,use_bias=False)
    def call(self,inp):
        return self.dense(inp)


class STE_embdding_layers(tf.keras.layers.Layer):
    def __init__(self,d_model,**kwargs):
        super(STE_embdding_layers,self).__init__(**kwargs)
        self.d_model = d_model
        self.dense = tf.keras.layers.Dense(self.d_model,use_bias=False)
    def call(self,te,se,if_te=True,if_se=True):
        '''

        Args:
            te: tensor (b,t,n,d)
            se: tensor(n,d)
            if_te: a boolen
            if_se: a boolen

        Returns:ste (b,t,n,d)

        '''
        B,T,N,D = get_shape(te)
        se_ = tf.tile(tf.expand_dims(tf.expand_dims(se,0),1),[B,T,1,1])
        if if_te and if_se:
          ste = se_+te
          return self.dense(ste)
        elif if_te and not if_se:
          return te
        elif if_se and not if_te:
          return se_

class QKV_projection(tf.keras.layers.Layer):
    def __init__(self,d_model):
        super().__init__()
        self.d_model = d_model
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    def call(self,quries,keys,values):
        '''
        original projection for multi head attention
        Args:
            quries: b,tq,n,d
            keys: b,tk,n,d
            values: b,tk,n,d

        Returns: three same shape tensors ->(b,t,n,d)

        '''
        return self.wq(quries),self.wk(keys),self.wv(values)

class Meta_projection(tf.keras.layers.Layer):
    def __init__(self,d_model):
        super().__init__()
        self.d1 = d_model/2
        assert d_model%2==0,'Dimension should be even!'
        self.d2 = d_model
        self.meta1 = [tf.keras.layers.Dense(d_model,activation=tf.keras.activations.relu,use_bias=True) for _ in range(3)]
        self.meta2 = [tf.keras.layers.Dense(tf.math.square(d_model),activation=tf.keras.activations.linear,use_bias=True) for _ in range(3)]
    def reshape(self,out):
        b,t,n,d = get_shape(out)
        return tf.reshape(out,[-1,t,n,self.d2,self.d2])
    def call(self,stq_q,stq_k,stq_v):
        '''
        Generating the weight matrix by meta-learning with shaoe (b,t,n,d,d)
        Args:
            stq_q: b,t,n,d
            stq_k: b,t,n,d
            stq_v: b,t,n,d

        Returns:w->(b,t,n,d,d)

        '''
        inp = [stq_q,stq_k,stq_v]
        w_list = []
        for i in range(3):
            out1_ = self.meta1[i](inp[i])
            out2_ = self.meta2[i](out1_)
            out_ = self.reshape(out2_)
            w_list.append(out_)
        return w_list

class Multi_head_temporal_attention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model  = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0,"The d_model is incorrect,Adjusted to an integer multiple of heads"
        self.qkv_proj = QKV_projection(d_model)
        self.meta_pro = Meta_projection(d_model)
        self.dense = tf.keras.layers.Dense(self.d_model,use_bias=False)
    def split_head(self,x):
        return tf.concat(tf.split(x,self.num_heads,axis=-1),axis=0)#-->(-1*h,t,n,d/h)
    def temporal_reshape(self,x):
        b,t,_,d = get_shape(x)
        x_  = tf.reshape(tf.transpose(x,[0,2,1,3]),[-1,t,d])
        return x_
    def concat_head(self,x):
        return tf.concat(tf.split(x,self.num_heads,axis=0),axis=-1)
    def call(self,queries,keys,values,tms,ste_q,ste_k,ste_v,if_ste,if_meta,if_future,if_spatial=False):
        '''
        Attention mechanisms in the temporal dimension
        Args:
            queries:b,t,n,d
            keys::b,t,n,d
            values::b,t,n,d
            ste_q:b,t,n,d
            ste_k:b,t,n,d
            ste_v:b,t,n,d
            if_ste:boolen
            if_meta:boolen
            if_future:boolen
            if_spatial:boolen

        Returns: (b,t,n,d),(bnh,tq,tk)

        '''
        if if_ste and not if_meta:
            q,k,v = self.qkv_proj(tf.add(queries+ste_q),tf.add(keys+ste_k),tf.add(values+ste_v))
        elif if_ste and if_meta:
            w_list = self.meta_pro(ste_q,ste_k,ste_v)
            wq,wk,wv = w_list
            q = meta_guide(queries,wq)
            k = meta_guide(keys,wk)
            v = meta_guide(values,wv)
        elif not if_ste:
            q,k,v = self.qkv_proj(queries,keys,values)#->(b,tk,n,d)
        b, t, n, d = get_shape(q)
        q,k,v = list(map(self.temporal_reshape,[q,k,v]))#->(bn,t,d)
        q_,k_,v_ = list(map(self.split_head,[q,k,v]))#->(bnh,t,d/h)
        out_ ,att = scale_dot_product_attention(q_,k_,v_,tm=tms,if_future=if_future,if_spatial=if_spatial)#-->(bnh,tq,d),(bnh,t,t)
        out_ = self.concat_head(out_)#->(bn,t,d)
        out_ = tf.transpose(tf.reshape(out_,(-1,n,t,d)),[0,2,1,3])
        out_ = self.dense(out_)
        return out_,att

class Multi_head_spatial_attention(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.d_model  = d_model
        self.num_heads = num_heads
        assert d_model%num_heads==0,"The d_model is incorrect,Adjusted to an integer multiple of heads"
        self.qkv_proj = QKV_projection(d_model)
        self.meta_pro = Meta_projection(d_model)
        self.dense = tf.keras.layers.Dense(self.d_model,use_bias=False)
    def split_head(self,x):
        return tf.concat(tf.split(x,self.num_heads,axis=-1),axis=0)#-->(-1*h,t,n,d/h)
    def spatial_reshape(self,x):
        b,_,n,d = get_shape(x)
        return tf.reshape(x,(-1,n,d))
    def concat_head(self,x):
        return tf.concat(tf.split(x,self.num_heads,axis=0),axis=-1)
    def call(self,queries,keys,values,tms,ste_q,ste_k,ste_v,if_ste,if_meta,if_future=False,if_spatial=True):
        '''
        Attention mechanisms in the spatial dimension
        Args:
            queries: b,t,n,d
            keys: b,t,n,d
            values: b,t,n,d
            tms: m,n,n
            ste_q: b,t,n,d
            ste_k: b,t,n,d
            ste_v: b,t,n,d
            if_ste: boolen
            if_meta: boolen
            if_future: boolen
            if_spatial: boolen

        Returns:(b,t,n,d),(bth,n,n)

        '''
        outs = []
        atts = []
        tms = tf.cast(tms,tf.float32)
        b, t, n, d = get_shape(queries)
        for num in range(tms.shape[0]):#tms->(m,n,n)
            if if_ste and not if_meta:
                q,k,v = self.qkv_proj(tf.add(queries+ste_q),tf.add(keys+ste_k),tf.add(values+ste_v))
            elif if_ste and if_meta:
                w_list = self.meta_pro(ste_q,ste_k,ste_v)
                wq,wk,wv = w_list
                q = meta_guide(queries,wq)
                k = meta_guide(keys,wk)
                v = meta_guide(values,wv)
            elif not if_ste:
                q,k,v = self.qkv_proj(queries,keys,values)#->(b,tk,n,d)
            q, k, v = list(map(self.spatial_reshape, [q, k, v]))  # ->(bt,n,d)
            q_, k_, v_ = list(map(self.split_head, [q, k, v]))  # ->(bth,n,d/h)
            out_, att = scale_dot_product_attention(q_, k_, v_, tm=tms[num], if_future=if_future, if_spatial=if_spatial)
            atts.append(att)
            out_ = self.concat_head(out_)#->(bt,n,d)
            outs.append(out_)
        out = tf.transpose(tf.convert_to_tensor(outs),[1,2,3,0])#->(m,bt,n,d)->(bt,n,d,m)
        out = tf.reshape(out,(-1,t,n,d*tms.shape[0]))#(b,t,n,dm)
        atts = tf.convert_to_tensor(atts)#->(m,bht,n,n)
        out = self.dense(out)#->(b,t,n,d)
        return out,atts

class FFN(tf.keras.layers.Layer):
    def __init__(self,d_model):
        super().__init__()
        self.d1 = d_model/4
        assert d_model%4==0,"The d should be divisible by 4"
        self.d2 = d_model
        self.dense1 = tf.keras.layers.Dense(self.d1,activation=tf.keras.activations.relu,use_bias=False)
        self.dense2 = tf.keras.layers.Dense(self.d2,activation=tf.keras.activations.linear,use_bias=False)
    def call(self,inp):
        '''
        dimension transformation
        Args:
            inp:(b,t,n,d)

        Returns:(b,t,n,d)

        '''
        out_ = self.dense1(inp)
        out_ = self.dense2(out_)
        return out_

class Encoderlayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,rate=hp.drop_rate,b_layers=hp.num_begin_blocks,m_layers=hp.num_medium_blocks,t_layers=hp.num_end_blocks):
        super().__init__()
        self.b_layers = b_layers
        self.m_layers = m_layers
        self.t_layers = t_layers
        #Bottom layer
        self.mtas1 = [Multi_head_temporal_attention(d_model,num_heads) for _ in range(b_layers)]
        self.dropouts1_0 = [tf.keras.layers.Dropout(rate) for _ in range(b_layers) ]
        self.layernorms1_0 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(b_layers)]
        self.msas1 = [Multi_head_spatial_attention(d_model,num_heads) for _ in range(b_layers)]
        self.dropouts1_1 = [tf.keras.layers.Dropout(rate) for _ in range(b_layers)]
        self.layernorms1_1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(b_layers)]
        # Middle layer
        self.mtas2 = [Multi_head_temporal_attention(d_model, num_heads) for _ in range(m_layers)]
        self.dropouts2_0 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_0 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        self.msas2 = [Multi_head_spatial_attention(d_model, num_heads) for _ in range(m_layers)]
        self.dropouts2_1 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        self.ffns1 = [FFN(d_model) for _ in range(m_layers)]
        self.dropouts2_2 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        # Top layer
        self.ffns2 = [FFN(d_model) for _ in range(t_layers)]
        self.dropouts3_0 = [tf.keras.layers.Dropout(rate) for _ in range(t_layers)]
        self.layernorms3_0 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(t_layers)]
    def call(self,queries,keys,values,tms,ste_q,ste_k,ste_v,if_ste,if_meta,training=True):
        # Bottom layer
        #(bnh,t,t)
        for i in range(self.b_layers):
            out_, att_t_botte_e = self.mtas1[i](queries,keys,values,None,ste_q,ste_k,ste_v,if_ste,if_meta,if_future=False,if_spatial=False)
            out_ = self.dropouts1_0[i](out_,training=training)
            out_ = self.layernorms1_0[i](queries+out_)
            #(m,bht,n,n)
            queries=keys=values=out_
            out_, att_s_botte_e = self.msas1[i](queries,keys,values,tms, ste_q, ste_k, ste_v, if_ste, if_meta, if_future=False,if_spatial=True)
            out_ = self.dropouts1_1[i](out_, training=training)
            out_ = self.layernorms1_1[i](queries + out_)
            queries = keys = values = out_
        # Middle layer
        for i in range(self.m_layers):
            out_, att_t_middle_e = self.mtas2[i](queries, keys, values,None, ste_q, ste_k, ste_v, if_ste, if_meta, if_future=False,if_spatial=False)
            out_ = self.dropouts2_0[i](out_, training=training)
            out_ = self.layernorms2_0[i](queries + out_)
            queries = keys = values = out_
            out_, att_s_middle_e = self.msas2[i](queries, keys, values, tms, ste_q, ste_k, ste_v, if_ste, if_meta, if_future=False,if_spatial=True)
            out_ = self.dropouts2_1[i](out_, training=training)
            out1 = self.layernorms2_1[i](queries + out_)
            out_ = self.ffns1[i](out1)
            out_ = self.dropouts2_2[i](out_, training=training)
            out2 = self.layernorms2_2[i](out_+out1)
            queries = keys = values = out_
        for i in range(self.t_layers):
            # Top layer
            out_ = self.ffns2[i](out2)
            out_ = self.dropouts3_0[i](out_, training=training)
            out_ = self.layernorms3_0[i](out_ + out2)
            out2 = out_
        return out_,att_t_botte_e,att_s_botte_e,att_t_middle_e,att_s_middle_e#only the last layer are reversed

class Encoder(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,rate=hp.drop_rate,b_layers=hp.num_begin_blocks,m_layers=hp.num_medium_blocks,t_layers=hp.num_end_blocks):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        #embeeding layers
        self.project = Projection(d_model)
        self.te = Temporal_embedding_layer(d_model,hp.input_len)
        self.se = Spatial_embedding_layers(d_model)
        self.ste_embedding = STE_embdding_layers(d_model)
        #enc_layers
        self.encoder_layer = Encoderlayer(d_model,num_heads,rate=rate,b_layers=b_layers,m_layers=m_layers,t_layers=t_layers)

    def call(self,x,n2v,tms,if_te=True,if_se=True,if_ste=True,if_meta=True,training=True):
        enc_inp = self.project(x)
        te_inp = self.te(x)
        se_inp = self.se(n2v)
        stes = self.ste_embedding(te_inp,se_inp,if_te,if_se)
        ste_q = ste_k = ste_v=stes
        queries=keys=values = enc_inp
        enc_out,att_t_botte_e,att_s_botte_e,att_t_middle_e,att_s_middle_e = self.encoder_layer(queries,keys,values,tms,ste_q,ste_k,ste_v,if_ste,if_meta,training=training)
        return enc_out,att_t_botte_e,att_s_botte_e,att_t_middle_e,att_s_middle_e,stes


class Decoderlayer(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,rate=hp.drop_rate,b_layers=hp.num_begin_blocks,m_layers=hp.num_medium_blocks,t_layers=hp.num_end_blocks):
        super().__init__()
        self.b_layers = b_layers
        self.m_layers = m_layers
        self.t_layers = t_layers
        #Bottom layer
        self.mtas1 = [Multi_head_temporal_attention(d_model,num_heads) for _ in range(b_layers)]
        self.dropouts1_0 = [tf.keras.layers.Dropout(rate) for _ in range(b_layers) ]
        self.layernorms1_0 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(b_layers)]
        self.msas1 = [Multi_head_spatial_attention(d_model,num_heads) for _ in range(b_layers)]
        self.dropouts1_1 = [tf.keras.layers.Dropout(rate) for _ in range(b_layers)]
        self.layernorms1_1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(b_layers)]
        self.mtias1 = [Multi_head_temporal_attention(d_model,num_heads) for _ in range(b_layers)]
        self.dropouts1_2 = [tf.keras.layers.Dropout(rate) for _ in range(b_layers)]
        self.layernorms1_2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(b_layers)]
        # Middle layer
        self.mtas2 = [Multi_head_temporal_attention(d_model, num_heads) for _ in range(m_layers)]
        self.dropouts2_0 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_0 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        self.msas2 = [Multi_head_spatial_attention(d_model, num_heads) for _ in range(m_layers)]
        self.dropouts2_1 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_1 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        self.mtias2 = [Multi_head_temporal_attention(d_model, num_heads) for _ in range(m_layers)]
        self.dropouts2_2 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_2 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        self.ffns1 = [FFN(d_model) for _ in range(m_layers)]
        self.dropouts2_3 = [tf.keras.layers.Dropout(rate) for _ in range(m_layers)]
        self.layernorms2_3 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(m_layers)]
        # Top layer
        self.ffns2 = [FFN(d_model) for _ in range(t_layers)]
        self.dropouts3_0 = [tf.keras.layers.Dropout(rate) for _ in range(t_layers)]
        self.layernorms3_0 = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(t_layers)]
    def call(self,queries,keys,values,enc_memory,tms,ste_enc,ste_q,ste_k,ste_v,if_ste,if_meta,training=True):
        # Bottom layer
        #(bnh,t,t)
        for i in range(self.b_layers):
            out_, att_t_botte_d = self.mtas1[i](queries,keys,values,None,ste_q,ste_k,ste_v,if_ste,if_meta,if_future=True,if_spatial=False)
            out_ = self.dropouts1_0[i](out_,training=training)
            out_ = self.layernorms1_0[i](queries+out_)
            #(m,bht,n,n)
            queries=out_
            out_, att_s_botte_d = self.msas1[i](out_,out_,out_,tms, ste_q, ste_k, ste_v, if_ste, if_meta, if_future=False,if_spatial=True)
            out_ = self.dropouts1_1[i](out_, training=training)
            out_ = self.layernorms1_1[i](queries + out_)
            queries = out_
            out_, att_ti_botte_d = self.mtias1[i](out_, enc_memory, enc_memory, None, ste_q, ste_enc, ste_enc, if_ste, if_meta,
                                                if_future=False, if_spatial=False)
            out_ = self.dropouts1_2[i](out_, training=training)
            out_ = self.layernorms1_2[i](queries + out_)
            queries = keys = values = out_
        # Middle layer
        for i in range(self.m_layers):
            out_, att_t_middle_d = self.mtas2[i](queries, keys, values,None, ste_q, ste_k, ste_v, if_ste, if_meta, if_future=True,if_spatial=False)
            out_ = self.dropouts2_0[i](out_, training=training)
            out_ = self.layernorms2_0[i](queries + out_)
            queries = out_
            out_, att_s_middle_d = self.msas2[i](out_, out_,out_, tms, ste_q, ste_k, ste_v, if_ste, if_meta, if_future=False,if_spatial=True)
            out_ = self.dropouts2_1[i](out_, training=training)
            out_ = self.layernorms2_1[i](queries + out_)
            queries = out_
            out_, att_ti_middle_d = self.mtias2[i](out_, enc_memory, enc_memory, None, ste_q, ste_enc, ste_enc, if_ste,
                                                  if_meta,if_future=False, if_spatial=False)
            out_ = self.dropouts2_2[i](out_, training=training)
            out1 = self.layernorms2_2[i](queries + out_)
            out_ = self.ffns1[i](out1)
            out_ = self.dropouts2_3[i](out_, training=training)
            out2 = self.layernorms2_3[i](out_+out1)
            queries = keys = values = out_
        for i in range(self.t_layers):
            # Top layer
            out_ = self.ffns2[i](out2)
            out_ = self.dropouts3_0[i](out_, training=training)
            out_ = self.layernorms3_0[i](out_ + out2)
            out2 = out_
        return out_,att_t_botte_d,att_s_botte_d,att_ti_botte_d,att_t_middle_d,att_s_middle_d,att_ti_middle_d#only the last layer are reversed

class Decoder(tf.keras.layers.Layer):
    def __init__(self,d_model,num_heads,rate=hp.drop_rate,b_layers=hp.num_begin_blocks,m_layers=hp.num_medium_blocks,t_layers=hp.num_end_blocks):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        #embeeding layers
        self.project = Projection(d_model)
        self.te = Temporal_embedding_layer(d_model,hp.output_len)
        self.se = Spatial_embedding_layers(d_model)
        self.ste_embedding = STE_embdding_layers(d_model)
        #dec_layers
        self.decoder_layer = Decoderlayer(d_model,num_heads,rate=rate,b_layers=b_layers,m_layers=m_layers,t_layers=t_layers)

    def call(self,x,n2v,tms,enc_memory,ste_enc,if_te=True,
             if_se=True,if_ste=True,if_meta=True,training=True):
        dec_inp = self.project(x)
        te_inp = self.te(x)
        se_inp = self.se(n2v)
        stes = self.ste_embedding(te_inp,se_inp,if_te,if_se)
        ste_q = ste_k = ste_v=stes
        queries=keys=values = dec_inp
        dec_out,att_t_botte_d,att_s_botte_d,att_ti_botte_d,att_t_middle_d,att_s_middle_d,att_ti_middle_d = self.decoder_layer(queries,keys,values,enc_memory,tms,
                                                                                               ste_enc,ste_q,ste_k,ste_v,if_ste,if_meta,training=training)
        return dec_out,att_t_botte_d,att_s_botte_d,att_ti_botte_d,att_t_middle_d,att_s_middle_d,att_ti_middle_d

