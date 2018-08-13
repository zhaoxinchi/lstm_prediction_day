# -*- coding: utf-8 -*-
from __future__ import print_function
import warnings
import os
import time
from keras.models import load_model
from load_testdata import *
from load_traindata import *
from plot_results import *
from build_model import *
from predict import *
from load_predict import *
import pandas as pd
from loadd import *

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':
    global_start_time = time.time()
    epochs = 1
    seq_len = 120

    print('> Loading data... ')
    ll = open('testdata.csv', 'r+').read()
    l = ll.split('\n')[:-1]
    flag=len(l)-271

    # 读取训练数据和测试数据
    # X_train, y_train, X_test, y_test,y_test_true = load_data('dataset.csv', seq_len, True)
    X_train, y_train = load_traindata('dataset.csv','train_label.csv', seq_len, True)
    # xtest, ytest, xtest_true, ytest_true,gap,spring = load_testdata('testdata.csv', 'test_label.csv',seq_len, True,151,200)
    xtest,xtest_true,gap,spring=load_predict('testdata.csv', 'test_label.csv',seq_len, True,151+flag,366)
    # print('X_train shape:', X_train.shape)
    # print('y_train shape:', y_train.shape)
    # print('xtest shape:', xtest.shape)
    # print('ytest shape:', ytest.shape)
    # print('> Data Loaded. Compiling...')

    # 训练模型
    # model = build_model([2, 120, 240, 1])
    # model.fit(X_train,y_train,batch_size=512,nb_epoch=epochs,validation_split=0.05)
    # model.save('lstm_test0802.h5')

    # 加载模型
    model = load_model('lstm_test0802.h5')
    # 预测
    point_by_point_predictions = predict_point_by_point(model, xtest)
    xtest_true = xtest_true.astype('float32')
    point_by_point_predictions = point_by_point_predictions.astype('float32')
    # 反归一化
    predict_true = []
    for i in range(len(point_by_point_predictions)):
        factor = (point_by_point_predictions[i] + 1) * xtest_true[i][0]
        predict_true.append(factor)

    if len(predict_true)>=spring[0]:
        predict_true[spring] = predict_true[spring] * 0.6
        predict_true[spring[0] - 1] = predict_true[spring[0] - 1] * 0.6


    count=0
    while (count<6):
        xtest, xtest_true, gap, spring = loadd('testdata.csv', 'test_label.csv', seq_len, True, 151+flag, 366, predict_true)
        point_by_point_predictions = predict_point_by_point(model, xtest)
        xtest_true = xtest_true.astype('float32')
        point_by_point_predictions = point_by_point_predictions.astype('float32')
        # 反归一化
        predict_true = []
        for i in range(len(point_by_point_predictions)):
            factor = (point_by_point_predictions[i] + 1) * xtest_true[i][0]
            predict_true.append(factor)

        count=count+1


    # plt.plot(predict_true)
    # plt.show()
    # print(predict_true)
    save=pd.DataFrame(data=predict_true)
    save.to_csv('prediction721.csv')


    # plot_results(predict_true, ytest_true, gap)  # 正常值数据显示





