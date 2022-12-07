# @Time    : 2022/6/9 11:22
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : frameworks
# @Project Name :metacode

import tensorflow as tf
from layers import *
from Hyperparameters import Hyperparameters as hp

class STGMT(tf.keras.Model):
    def __init__(self,d_model,num_heads,rate=hp.drop_rate):
        super().__init__()
        self.encoder = Encoder(d_model,num_heads,rate)
        self.decoder = Decoder(d_model,num_heads,rate)
        self.output_layer = tf.keras.layers.Dense(hp.out_units)
    def call(self,enc_inp,dec_inp,n2v,tms,if_te=True,if_se=True,if_ste=True,if_meta=True,training=True):
        enc_out,att_t_botte_e,att_s_botte_e,att_t_middle_e,att_s_middle_e,stes = self.encoder(enc_inp,n2v,tms,if_te,
                                                                                              if_se,if_ste,if_meta,training)
        enc_memory = enc_out
        ste_enc = stes
        dec_inp = tf.cast(dec_inp,tf.float32)
        dec_inp = tf.concat([enc_inp[:,-1:,:,:],dec_inp[:,:-1,:,:]],axis=1)#right shifted
        dec_out, att_t_botte_d, att_s_botte_d, att_ti_botte_d, att_t_middle_d, att_s_middle_d, att_ti_middle_d = \
            self.decoder(dec_inp,n2v,tms,enc_memory,ste_enc,if_te,
             if_se,if_ste,if_meta,training)
        final_output = self.output_layer(dec_out)
        enc_atts = [att_t_botte_e,att_s_botte_e,att_t_middle_e,att_s_middle_e]
        dec_atts = [att_t_botte_d, att_s_botte_d, att_ti_botte_d, att_t_middle_d, att_s_middle_d, att_ti_middle_d]
        return final_output,enc_atts,dec_atts




class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):  # lr setting
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)



# def main():
#     st_model = STGMT(16, 4)
#     enc_inp = tf.random.uniform((2, hp.input_len, 14, 1))
#     dec_inp = tf.random.uniform((2, hp.output_len, 14, 1))
#     n2v = tf.random.uniform((14, 64))
#     tms = tf.random.uniform((2, 14, 14))
#     final_output, enc_atts, dec_atts = st_model(enc_inp, dec_inp, n2v, tms)
#     print(final_output.shape)
#     print(st_model.summary())
# if __name__ == '__main__':
#     main()



