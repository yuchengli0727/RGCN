from torch.utils.data import DataLoader
from traffic_dataset import LoadData
import torch.nn as nn
import torch
import os
import time
import h5py
import numpy as np
import torch.optim as optim
from utils import Evaluation
from utils import visualize_result
import matplotlib.pyplot as plt

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


#   注意每一个变量是tensor还是numpy，维度是多少，每个维度代表什么
class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()
        self.gru = nn.GRU(6, 307, batch_first=True)
        self.hidden = torch.zeros(1, 64, 307)
        self.fc = nn.Linear(307, 1)
        self.relu = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1

        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C] -- [64, 307, 6]

        output_3 = self.linear_2(output_2)
        output_3 = self.act(torch.matmul(graph_data, output_3))  # [B, N, 1, Out_C] -- [64, 307, 6]

        output_gru, _ = self.gru(output_3)
        out = self.fc(self.relu(output_gru))    #[64, 307, 1]
        #print(out.shape)

        return out

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    normdata = np.load("pemstest.npz")
    normdata = normdata['arr_0'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]
    max_data = np.max(normdata, 1, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
    min_data = np.min(normdata, 1, keepdims=True)
    mid = min_data
    base = max_data - min_data
    mid = mid.squeeze(2)
    base = base.squeeze(2)
    # Loading Dataset
    train_data = LoadData(data_path=["PeMS04.csv", "pemstest.npz"], num_nodes=307, divide_days=[3, 2],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)

    test_data = LoadData(data_path=["PeMS04.csv", "pemstest.npz"], num_nodes=307, divide_days=[3, 2],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8)

    # Loading Model
    my_net = GCN(in_c=6, hid_c=6, out_c=6)
   # my_net = GATNet(in_c=6 * 1, hid_c=6, out_c=1, n_heads=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Epoch = 200
    loss_value = []


    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover

            loss = criterion(predict_value, data["flow_y"].squeeze(3))

            epoch_loss += loss.item()


            loss.backward()

            optimizer.step()
        end_time = time.time()
        loss_value.append(1000 * epoch_loss / len(train_data))
        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))

    plt.plot(loss_value)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.show()

    my_net.eval()
    with torch.no_grad():
        MAE, MAPE, RMSE, acu = [], [], [], []
        Target = np.zeros([307, 64]) # [N, 1, D]
        Predict = np.zeros_like(Target)  #[N, T, D]

        total_loss = 0.0
        for data in test_loader:

            predict_value = my_net(data, device).to(torch.device("cpu"))  # [64, 307, 1]

            loss = criterion(predict_value, data["flow_y"].squeeze(3))
            # data[flow_y]:[64, 307, 1, 1]      data[flow_y].squeeze(3):[64, 307, 1]

            total_loss += loss.item()

            predict_value = predict_value.numpy()
            target_value = data["flow_y"].squeeze(3).numpy()

            predict_value = predict_value * base + mid
            target_value = target_value * base + mid

            predict_value_graph = predict_value.transpose([1, 0, 2]).squeeze(2)
            target_value_graph = target_value.transpose([1, 0, 2]).squeeze(2)

            Predict = np.concatenate([Predict, predict_value_graph], axis=1)
            Target = np.concatenate([Target, target_value_graph], axis=1)

            performance = Evaluation.total(target_value, predict_value)

            MAE.append(performance[0])
            MAPE.append(performance[1])
            RMSE.append(performance[2])
            acu.append(performance[3])

        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

    print("Performance:  MAE {:2.2f}    mape{:2.2f}%    rmse{:2.2f}   acu{:2.2f}".format(np.mean(MAE), np.mean(MAPE * 100), np.mean(RMSE), np.mean(acu)))

    y_true = Target[0, 100:]
    y_pred = Predict[0, 100:]
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.legend(["prediction", "target"], loc="upper right")
    plt.xlabel("Time")
    plt.ylabel("Traffic flow")
    #plt.xticks(np.arange(0, 7, 1), ["2018-2-10","2018-2-11","2018-2-12","2018-2-13","2018-2-14","2018-2-15","2018-2-16"])
    plt.show()


if __name__ == '__main__':
    main()

