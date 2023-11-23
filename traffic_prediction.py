import os
import time
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from traffic_dataset import LoadData
from utils import Evaluation
from utils import visualize_result
# from gat import GRU

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

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)   # inc 表示输入的个数， hidc表示神经元的数量（输出数量） 权重矩阵[hidc, inc] 偏移矩阵[hidc]
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D] [64, 207, 6, 1]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1 view用来改变tensor的形状 torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度

        output_1 = self.linear_1(flow_x)  # [B, N, hid_C] 线性层，W - [] - [6 * 6]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, Out_C]

        return output_2.unsqueeze(2)    # 在维度2上增加一个维度，[B, N, Out_C] -- [B, N, 1, Out_C]

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)   # 生成一个对角元素为1的矩阵
        graph_data += matrix_i  # A~ [N, N] 含自连接的邻接矩阵

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N] 对每一行元素相加，且不保留维度 D~
        degree_matrix = degree_matrix.pow(-1)   # 对每个元素求倒数 D~（-1）
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]  取矩阵的对角线元素 如果只有一维，则生成N*N的矩阵，对角线元素为原本的一维矩阵  D~（-1）

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A) 这个等价于\hat A = D_{-1/2}*A*D_{-1/2}


class Baseline(nn.Module):
    def __init__(self, in_c, out_c):
        super(Baseline, self).__init__()
        self.layer = nn.Linear(in_c, out_c)

    def forward(self, data, device):
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output = self.layer(flow_x)  # [B, N, Out_C], Out_C = D

        return output.unsqueeze(2)  # [B, N, 1, D=Out_C]


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Loading Dataset
    train_data = LoadData(data_path=["PeMS04.csv", "PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

    test_data = LoadData(data_path=["PeMS04.csv", "PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=32)

    # Loading Model
    my_net = GCN(in_c=6, hid_c=6, out_c=1)
    # my_net = GRU(input_dim=6, hidden_dim=6, output_dim=1, n_layers=1)
    # my_net = GATNet(in_c=6 * 1, hid_c=6, out_c=1, n_heads=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Epoch = 1

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

            loss = criterion(predict_value, data["flow_y"])

            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))

    # Test Model
    my_net.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE = [], [], []
        Target = np.zeros([307, 1, 1]) # [N, 1, D]
        Predict = np.zeros_like(Target)  #[N, T, D]  zero_like(w)可以创建一个与w大小相同的全零矩阵

        total_loss = 0.0
        for data in test_loader:

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D]  -> [1, N, B(T), D]

            loss = criterion(predict_value, data["flow_y"])

            total_loss += loss.item()

            predict_value = predict_value.transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]
            target_value = data["flow_y"].transpose(0, 2).squeeze(0)  # [1, N, B(T), D] -> [N, B(T), D] -> [N, T, D]

            performance, data_to_save = compute_performance(predict_value, target_value, test_loader)

            Predict = np.concatenate([Predict, data_to_save[0]], axis=1)
            Target = np.concatenate([Target, data_to_save[1]], axis=1)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])

        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

    print("Performance:  MAE {:2.2f}    {:2.2f}%    {:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE)))

    Predict = np.delete(Predict, 0, axis=1)
    Target = np.delete(Target, 0, axis=1)

    result_file = "GAT_result.h5"
    file_obj = h5py.File(result_file, "w")

    file_obj["predict"] = Predict
    file_obj["target"] = Target


def compute_performance(prediction, target, data):
    try:
        dataset = data.dataset  # dataloader
    except:
        dataset = data  # dataset

    prediction = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], prediction.numpy())
    target = LoadData.recover_data(dataset.flow_norm[0], dataset.flow_norm[1], target.numpy())

    mae, mape, rmse = Evaluation.total(target.reshape(-1), prediction.reshape(-1))

    performance = [mae, mape, rmse]
    recovered_data = [prediction, target]

    return performance, recovered_data


if __name__ == '__main__':
    main()
    visualize_result(h5_file="GAT_result.h5",
                     nodes_id=120,
                     time_se=[0, 24 * 12 * 2],
                     visualize_file="gat_node_120")

