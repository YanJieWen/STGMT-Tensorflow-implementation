# @Time    : 2022/6/9 11:23
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : gen_batch
# @Project Name :metacode
from utils_ import *
def gen_batch(datas,batch_size):
    len_data = len(datas)
    idx = np.arange(len_data)
    np.random.shuffle(idx)
    for i in range(0,len_data,batch_size):
        start_idx = i
        end_idx = i+batch_size
        if end_idx>len_data:
            end_idx = len_data
        slc = idx[start_idx:end_idx]
        yield datas[slc]