import numpy as np
from normalise_windows import *

def load_traindata(filename,filename_label, seq_len, normalise_window):  # 加载数据并划分数据集
    f = open(filename, 'r+').read()  # 读取数据文件
    l=open(filename_label,'r+').read()  #读取标签

    # 以换行分割数据转化为list
    data = f.split('\n')[:-1]
    label = l.split('\n')[:-1]

    print('data len:', len(data))  # 输出数据的长度1827
    print('label len:',len(label))
    print('sequence len:', seq_len)

    sequence_length = seq_len + 1

    result = []
    for index in range(len(data) - sequence_length+1):
        result.append(data[index: index + sequence_length])  # 得到长度为seq_len+1的向量，最后一个作为y

    # print('result len:', len(result))
    print('train result shape:', np.array(result).shape)      #1707,121

    if normalise_window:
        result = normalise_windows(result)

    print('normalise_windows result shape:', np.array(result).shape)        #1707,121

    # 客流量组合label成为三维输入
    result_dim3 = []
    for i in range(len(data)-seq_len):
        result_dim2 = []
        for j in range(seq_len):
            result_dim2.append([result[i][j],label[i+j]])
        result_dim3.append(result_dim2)

    print('dim3:',np.array(result_dim3).shape)      #1707,120,2

    train = np.array(result)
    # x_train1 = train[:, :-1]
    # x_train=np.zeros((1707,121))
    y_train = train[:, -1]

    # for t in range(1707):
    #     x_train[t] = np.append(x_train1[t], label[t+120])


    # reshape X to be [samples, time steps, features]
    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 转化为三维输入
    x_train=np.array(result_dim3)

    return [x_train, y_train]