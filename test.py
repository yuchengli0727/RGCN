'''
import numpy as np
import matplotlib.pyplot as plt
data = np.load("PeMS04.npz")
flow = data["data"].transpose([1, 0, 2])[:, :, 0]
flow_node = flow[0]
#print(flow_node)

max_data = np.max(flow_node)
min_data = np.min(flow_node)
noraml_flow = (flow_node - min_data) / (max_data - min_data)
#plt.plot(noraml_flow)
#plt.show()
train_x = []
'''


import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
def scale(X: np.ndarray, y: np.ndarray):
    """最大最小值归一化"""
    x_sc = MinMaxScaler()
    x_train_scaled = x_sc.fit_transform(X)
    y_sc = MinMaxScaler()
    y_train_scaled = y_sc.fit_transform(y.reshape(-1, 1))
    return x_train_scaled, y_train_scaled, x_sc, y_sc


def lookback(x, y, period=24, days=7):
    """
    define lookback period
    构建特征：这里的特征是多维数组
    基本思想：当天未来一天的预测值跟过去days天的数据相关
    :param x: 表示特征
    :param y: 表示标签
    :param period:数据周期
    :param period:选择过去多少天作为特征
    :return: 用于lstm建模的输入特征和标签
    """
    # 构建0数组，用于后面赋值数据
    inputs = np.zeros((int(x.shape[0] / period) - days, period * (days - 1), x.shape[1]))
    labels = np.zeros((int(x.shape[0] / period) - days, period))
    # print(inputs.shape, labels.shape)

    for i in range(period * days, x.shape[0] - period, period):
        """顺移
        0---0:90;
        1---1:91;......
        """
        # print(i, period, (i - period * 2) / 96, y[i:i + period].shape)
        inputs[int((i - period * days) / period)] = x[i - period * days:i - period]
        labels[int((i - period * days) / period)] = y[i:i + period].reshape(-1, )

    # print(i)
    inputs = inputs.reshape(-1, period * (days - 1), x.shape[1])
    labels = labels.reshape(-1, period)
    # print(x.shape, y.shape)
    # print(inputs.shape, labels.shape)
    return inputs, labels

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'device is {device}')

class GRUnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUnet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

def train(train_loader, learn_rate, hidden_dim=256, EPOCHS=10, model_type='GRU'):
    # setting common hyperparameters
    # 这里求快，随便设置了epoch次数，可根据模型结果调整
    input_dim = next(iter(train_loader))[0].shape[2]
    output_dim = period
    n_layers = 1
    # instantiating the models
    if model_type == 'GRU':
        model = GRUnet(input_dim, hidden_dim, output_dim, n_layers)

    model.to(device)

    #  defining loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.train()
    print('starting training of {} model'.format(model_type))

    epoch_times = []
    # start training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.process_time()
        h = model.init_hidden(batch_size)
        avg_loss = 0
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == 'GRU':
                h = h.data
            else:
                h = tuple([e.data for e in h])
            model.zero_grad()

            out, h = model(x.to(device).float(), h)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            if counter % 200 == 0:
                print('epoch {}.... step: {}/{}...average loss for epoch: {}'.format(
                    epoch, counter, len(train_loader), avg_loss / counter))
        if epoch % 10 == 0:
            current_time = time.process_time()
            print("Epoch {}/{} Done, Total Loss: {}".format(epoch, EPOCHS, avg_loss / len(train_loader)))

            print('time elapsed for epoch: {} seconds'.format(str(current_time - start_time)))
            epoch_times.append(current_time - start_time)
    print('total traning time: {} seconds'.format(str(sum(epoch_times))))
    return model

def evaluate(model, test_x, test_y, label_scalers=None):
    model.eval()
    start_time = time.process_time()
    inp = torch.from_numpy(np.array(test_x))
    labs = torch.from_numpy(np.array(test_y))
    h = model.init_hidden(inp.shape[0])
    out, h = model(inp.to(device).float(), h)

    if label_scalers is not None:
        outputs = (label_scalers.inverse_transform(out.cpu().detach().numpy()).reshape(-1))
        targets = (label_scalers.inverse_transform(labs.numpy()).reshape(-1))
    else:
        outputs = (out.cpu().detach().numpy()).reshape(-1)
        targets = (labs.numpy()).reshape(-1)

    print('evaluation time: {}'.format(str(time.process_time() - start_time)))
    return outputs, targets


def features(df):
    """提取df特征，这里与前面的lookback不同;
    lookback是用于模型,产生多维数组;
    这里是作用于单个记录;
    除了以下时间特征，还可以提取节假日特征、缺失值查找和填充;
    数据充足的话，还可添加气象数据、经济相关的数据等"""
    df['hour'] = df.index.hour
    df['weekday'] = df.index.weekday
    df['month'] = df.index.month
    df['is_weekend'] = df['weekday'].isin([5, 6]) * 1
period = 24
days = 7

"""数据类型、缺失情况(df.isna().sum())等，可绘图查看"""
if __name__ == '__main__':
    df = pd.read_csv('./AEP_hourly.csv', parse_dates=['Datetime'], index_col=0)
    features(df)

    # 分割训练集和测试集
    df_train = df[df.index.year < 2018]
    df_test = df[df.index.year >= 2018]

    # 特征标签归一化
    x_train_scaled, y_train_scaled, x_sc, y_sc = scale(X=df_train.values,
                                                       y=df_train['AEP_MW'].values)
    x_test_scaled = x_sc.transform(df_test.values)
    y_test_scaled = y_sc.transform(df_test['AEP_MW'].values.reshape(-1, 1))
    train_inputs, train_labels = lookback(x_train_scaled, y_train_scaled)
    test_inputs, test_labels = lookback(x_test_scaled, y_test_scaled)
    batch_size = period
    train_data = TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_labels))
    train_loader = DataLoader(train_data, shuffle=False, drop_last=True, batch_size=batch_size)
    lr = 0.001
 # gru 模型
    gru_model = train(train_loader, lr, model_type="GRU")
    gru_outputs, targets = evaluate(gru_model, test_inputs, test_labels, y_sc)
    # gru_outputs, targets = evaluate(gru_model, test_inputs, test_labels)
    gru_error = mean_squared_error(gru_outputs, targets)

    print('gru_error:\n', gru_error)

    print('\n\n')

