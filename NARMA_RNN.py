import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import sys

def NARMA10(Two, time_size, sigma):
    '''创建时间序列
    Two: ESN的预热时间
    T: ESN的预测时间
    sigma: 输入信号相关参数
    '''
    # 输入信号：包括预热时间
    zeta = 2*np.random.rand(Two+time_size)-1 # [-1, 1]均匀分布
    u = sigma*0.5*(zeta+1) # u in [0, sigma]
    
    # 输出信号
    z = np.zeros(Two+time_size)
    for t in range(10, Two+time_size):
        z[t] = 0.3*z[t-1] + 0.05*z[t-1]*np.sum(z[t-10:t]) + 1.5*u[t-1]*u[t-10] + 0.1
    
    return zeta, z


class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(
                    input_size, 
                    hidden_size, 
                    num_layers, 
                    batch_first=True
        )
        # for p in self.rnn.parameters():
        #     nn.init.normal_(p, mean=0.0, std=0.001)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        
        return out, hidden


# Generate NARMA-10 data and split into train and test sets
Two = int(1e3)
data_size = int(1e4)
sigma = 0.45
train_size = int(data_size * 0.8)
val_size = int(data_size * 0.1)
test_size = data_size - train_size - val_size
np.random.seed(0)
train_inputs, train_targets = NARMA10(Two, train_size, sigma)
np.random.seed(10)
val_inputs, val_targets = NARMA10(Two, val_size, sigma)
np.random.seed(0)
test_inputs, test_targets = NARMA10(Two, test_size, sigma)

# 实例化模型
window_in = 11
window_out = 1
start_index = max(window_in, 9+window_out)
model = MyRNN(input_size=1, hidden_size=50, output_size=window_out, num_layers=1)
criterion = nn.MSELoss()
lr = 0.01
optimizer = optim.SGD(model.parameters(), lr=lr)
num_epochs = 50


# 训练模型
train_losses, val_losses = [], []
nmse_val_last_epoch = 1
for epoch in range(num_epochs):
    
    train_hidden = None

    # 查看学习率
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    
    '''训练阶段'''
    # 数据索引
    index_vector = list(range(start_index, Two+train_size))
    # np.random.shuffle(index_vector) # 打乱

    # 训练
    model.train()
    running_loss = 0.0
    for i in index_vector:
        input = torch.Tensor(train_inputs[i-window_in : i].reshape(1, window_in, 1))
        target = torch.Tensor(train_targets[i-window_out+1 : i+1].reshape(1, -1))
        output, train_hidden = model(input, train_hidden) # [1, 10]

        train_hidden = train_hidden.data
        loss = criterion(output, target)

        optimizer.zero_grad() # 注意摆放位置
        loss.backward()
        running_loss += loss.item()

        # 梯度更新
        optimizer.step()
        # model_dict = model.state_dict()
        # for k, v in model.named_parameters():
        #     model_dict[k] -= lr * v.grad.detach().numpy()
        # model.load_state_dict(model_dict)

    train_losses.append(running_loss / len(index_vector))

    # if epoch > 0:
    #     nmse_val_last_epoch = nmse_val

    '''验证阶段'''
    val_hidden = None
    model.eval()
    with torch.no_grad():
        targets, outputs = np.zeros(Two+val_size), np.zeros(Two+val_size)
        for i in range(start_index, Two+val_size):
            val_input = torch.Tensor(val_inputs[i-window_in : i].reshape(1, window_in, 1))
            val_output, val_hidden = model(val_input, val_hidden)
            val_hidden = val_hidden.data
            targets[i] = val_targets[i]
            outputs[i] = val_output[0, -1].item()
        # NMSE
        mse_val = np.mean(np.square((targets[Two:] - outputs[Two:])))
        nmse_val = mse_val / np.var(targets[Two:])

        # if nmse_val > nmse_val_last_epoch:
        #     lr = lr * 0.75

    print(f'Epoch {epoch+1}: lr = {lr:.5f}, train loss = {train_losses[-1]:.5f}, val nmse = {nmse_val:.5f}')

# 测试模型
targets, outputs = np.zeros(Two+test_size), np.zeros(Two+test_size)
test_hidden = None
for i in range(start_index, Two+test_size):
    test_input = torch.Tensor(test_inputs[i-window_in : i].reshape(1, window_in, 1))
    test_output, test_hidden = model(test_input, test_hidden)
    test_hidden = test_hidden.data
    targets[i] = test_targets[i]
    outputs[i] = test_output[0, -1].item()

mse_test = np.mean(np.square((targets[Two:] - outputs[Two:])))
nmse_test = mse_test / np.var(targets[Two:])
print('NARMA10 NMSE(test): ', nmse_test)

import matplotlib.pyplot as plt
plt.plot(targets[Two:], 'r')
plt.plot(outputs[Two:], 'b')
plt.show()
