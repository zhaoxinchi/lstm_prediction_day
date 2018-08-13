from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time


# 创建模型
def build_model(layers):  # layers [2, 120, 240, 1]
    model = Sequential()

    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop") #优化器配置
    model.compile(loss='mse', optimizer='adam')
    print("Compilation Time : ", time.time() - start)
    return model