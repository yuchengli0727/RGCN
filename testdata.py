from torch.utils.data import DataLoader
from traffic_dataset import LoadData
import torch
import torch.nn as nn
import numpy as np

data = np.load("PeMS04.npz")
print(data.files)
data = data['data']
print(data.shape)




'''
train_data = LoadData(data_path=["PeMS04.csv", "PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                      time_interval=5, history_length=6,
                      train_mode="train")

#train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=32)

graph_data = train_data[0]["graph"]
flow_x = train_data[0]["flow_x"]
print(flow_x.shape)
N = graph_data.size(0)
matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)  # 生成一个对角元素为1的矩阵
graph_data += matrix_i  # A~ [N, N] 含自连接的邻接矩阵

degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]

degree_matrix2 = degree_matrix.pow(-1)

degree_matrix2 = torch.diag(degree_matrix2)  # [N, N]

final_graph2 = torch.mm(degree_matrix2, graph_data)
#print(final_graph2)
gru = nn.GRU(6, 307)
input, _ = gru(flow_x.permute(2, 0, 1).contiguous())
print(input)
'''