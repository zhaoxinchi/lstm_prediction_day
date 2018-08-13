import matplotlib.pyplot as plt



# 画图
def plot_results(predicted_data, true_data,j ):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.xticks([x for x in range(j)])  # x标记step设置为1
    plt.legend()  # 显示图例
    plt.show()
    # plt.savefig(filename+'.png')