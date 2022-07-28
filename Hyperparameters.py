# @Time    : 2022/6/9 11:21
# @Author  : Yanjie WEN
# @Institution : CSU & BUCEA
# @IDE : Pycharm
# @FileName : Hyperparameters
# @Project Name :metacode

#Train your own data, Liver Explosion Parameters are necessary
class Hyperparameters:
    #data
    data_04 = './data/PEMS04'
    data_08 = './data/PEMS08'
    data_nyc = './data/NYC'
    pkl_04 = 'PEM04.pkl'
    pkl_08 = 'PEM08.pkl'
    NYC_ = 'NYC.pkl'
    batch_size = 12#memory
    input_len = 48#inputting time sequence length
    output_len = 12#outputting time sequence length
    #graph embedding
    d_node = 64
    walk_len = 16
    num_walks = 100
    p = 0.3 #pems08->0.3.NYC->1
    q = 0.7 #if q<1 DFS; q>1 BFS; q=1 deepwalk pems08->0.7.NYC->2
    workers = 4
    eplision=0.1
    time_eplision = 0.3

    #model
    num_units =16#memory
    out_units = 1
    if_te = True#temproal embedding
    if_se = True#spatio embedding
    if_ste = True # spatiol&temproal embedding
    if_meta= True#meta learning
    drop_rate = 0.0001
    num_heads=4#memory
    num_begin_blocks=1#memory
    num_medium_blocks =1#memory
    num_end_blocks = 1#memory
    #train
    if_onlydist = False
    if_onlytimesim=False
    if_val = True
    lr = 0.0001
    logdir = 'logdir'
    num_epochs = 100
    valid_thresh = 50
    ckpt_path = './ckpt/weight'
    #test
    step_index = [2,5,8,11]#3,6,9,12