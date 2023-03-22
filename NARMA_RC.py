import numpy as np

import sys

def NARMA10(Two, time_size, sigma):
    '''创建时间序列
    Two: ESN的预热时间
    time_size: ESN的预测时间
    sigma: 输入信号相关参数
    '''
    # 输入信号：包括预热时间
    zeta = 2*np.random.rand(Two+time_size)-1 # [-1, 1]均匀分布
    u = sigma*0.5*(zeta+1) # u in [0, sigma]
    
    # 输出信号
    z = np.zeros(Two+time_size)
    for t in range(10, Two+time_size):
        z[t] = 0.3*z[t-1] \
             + 0.05*z[t-1]*np.sum(z[t-10:t]) \
             + 1.5*u[t-1]*u[t-10] + 0.1
    
    return zeta, z


##### Parameters for esn #####
N = 50
p = 0.5
pin = 0.1
iota = 0.1
rho = 0.9
#weight
p = 0.5
pin = 0.1
seed = 0
np.random.seed(seed)
win = (2*np.random.rand(N)-1) * (np.random.rand(N)<pin)
w = (2*np.random.rand(N,N)-1) * (np.random.rand(N,N)<p)
eig, eigv = np.linalg.eig(w)
w = w / np.max(np.abs(eig))


##### input #####
Two = int(1e3)
data_size = int(1e4)
# 划分数据集
train_size = int(data_size * 0.8)
test_size = int(data_size * 0.1)


# NARMA10
sigma = 0.45 # 为什么0.5就算不出来
np.random.seed(0)
zeta_train, z_train = NARMA10(Two, train_size, sigma)

# 训练ESN
x_train = np.zeros((N, Two+train_size)) # 节点初始状态
for t in range(1, Two+train_size):
    x_train[:, t] = np.sin(rho*w.dot(x_train[:, t-1]) + iota*win*zeta_train[t-1])
x_train_add = np.row_stack((x_train, np.ones(Two+train_size)))

alpha = 0
Wout = np.linalg.inv((x_train_add[:, Two:] @ x_train_add[:, Two:].T) + (alpha * np.identity(x_train_add[:, Two:].T.shape[1]))) @ (x_train_add[:, Two:] @ z_train[Two:])

zhat_train = np.dot(Wout, x_train_add)
mse_train = np.mean(np.square((z_train[Two:] - zhat_train[Two:])))
nmse_train = mse_train / np.var(z_train[Two:])
print('NARMA10 NMSE(train): ', nmse_train)


# 测试ESN
np.random.seed(0)
zeta_test, z_test = NARMA10(Two, test_size, sigma)
x_test = np.zeros((N, Two+test_size))
for t in range(1, Two+test_size):
    x_test[:, t] = np.sin(rho*w.dot(x_test[:, t-1]) + iota*win*zeta_test[t-1])
x_test_add = np.row_stack((x_test, np.ones(Two+test_size))) # 加上偏置
zhat_test = np.dot(Wout, x_test_add) # 预测值
mse_test = np.mean((z_test[Two:]-zhat_test[Two:])**2)
nmse_test = mse_test / np.var(z_test[Two:])
print('NARMA10 NMSE(test): ', nmse_test)

import matplotlib.pyplot as plt
plt.plot(z_test[Two:], 'r')
plt.plot(zhat_test[Two:], 'b')
plt.show()

