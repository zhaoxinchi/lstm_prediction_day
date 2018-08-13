import numpy as np
from normalise_windows import *


def load_testdata(filename,filename_label, seq_len, normalise_window,start,end):
    ft = open(filename, 'r+').read()  # 读取数据文件
    lt = open(filename_label,'r+').read()

    data = ft.split('\n')[:-1]  # 以换行分割数据转化为list
    label=lt.split('\n')[:-1]
    print('testdata len:', len(data))  # 输出数据的长度
    print('sequence len:', seq_len)

    # 寻找春节的标签
    spring=[]
    for i in range(len(label)):
        if label[i]=='-0.75':
            spring.append(i)

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length+1):
        result.append(data[index: index + sequence_length])

    print('test result shape:',np.array(result).shape)   #365,121
    x_test_true = np.array(result)[start:end, :-1]
    y_test_true = np.array(result)[start:end, -1]

    if normalise_window:
        result = normalise_windows(result)

    # 客流量组合label成为三维输入
    result_dim3 = []
    for i in range(len(data)-seq_len):
        result_dim2 = []
        for j in range(seq_len):
            result_dim2.append([result[i][j], label[i + j]])
        result_dim3.append(result_dim2)

    print('dim3:', np.array(result_dim3).shape)  # 365,120,2

    result = np.array(result)
    # x_test1 = result[:31, :-1]
    # x_test = np.zeros((31, 121))
    # for t in range(31):
    #     x_test[t] = np.append(x_test1[t], label[t+120])


    y_test = result[start:end, -1]
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    x_test=np.array(result_dim3)[start:end,:,:]
    gap=end-start
    spring[0]=spring[0]-seq_len
    return [x_test, y_test, x_test_true, y_test_true,gap,spring]