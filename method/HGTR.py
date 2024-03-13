import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear, to_hetero,SAGEConv


def otof(x,type,c,h): #输入x，输出四类[tensor,tensor,tensor,tensor]

    unique_types = torch.unique(type)  #1,2,3,4
    segmented_matrices = []
    for t in unique_types:
        # 创建布尔掩码，选择当前类型的像素
        mask = (type == t).unsqueeze(0).expand_as(x)
        # 使用布尔掩码提取当前类型的像素
        segmented_a = torch.masked_select(x, mask).view(c, -1).t()
        # 存储分割后的矩阵
        segmented_matrices.append(segmented_a)
    return segmented_matrices

def ftoo(segmented_matrices,type_tensor,c,h): #还原
    restored_a = torch.zeros(c, h, h).to(device)
    for i, matrix in enumerate(segmented_matrices):
        current_type = i+1  # 假设类型从1到4
        type_mask = (type_tensor == current_type).unsqueeze(0).expand_as(restored_a)
        # 将当前类型的矩阵还原到原始像素图的对应位置
        restored_a.masked_scatter_(type_mask, matrix.t().contiguous().view(c,-1))
    return restored_a

def get_type(x):#像素分类
    # thresholds = [0.25, 0.7]  # 设置分割阈值，将图像分为四类
    # thresholds = [0.1, 0.9]  # 设置分割阈值，将图像分为四类
    # thresholds = [0.3, 0.6]  # 设置分割阈值，将图像分为四类
    thresholds = [0.4, 0.5]  # 设置分割阈值，将图像分为四类

    pred = x  # 改
    h ,w= pred.shape
    node_types = torch.zeros(h, w, dtype=torch.long).to(device)  # 初始化节点类别信息
    node_types += (pred >= thresholds[0]).squeeze().long()
    node_types += (pred >= thresholds[1]).squeeze().long()
    node_types += 1  # 类别从 1 到 4
    return node_types

def get_edge(node_types,node_type1,node_type_2):
    src_nodes_2 = torch.nonzero(node_types == node_type_2).squeeze().to(device)   # 类别 2 的节点#计算所有3附近5个2的节点
    src_nodes_3 = torch.nonzero(node_types == node_type1).squeeze().to(device)   # 类别 3 的节点
    point_count = src_nodes_3.shape[0]
    # src_nodes = src_nodes_3[:, 0] * 128 + src_nodes_3[:, 1]
    # src_nodes = torch.repeat_interleave(src_nodes, 5)
    src_nodes = torch.repeat_interleave(torch.arange(0, point_count, 1), 5).to(device)
    dst_nodes = torch.tensor([], dtype=torch.long).to(device)
    for k in range(point_count):
        q = knn_search(src_nodes_2, src_nodes_3[k].unsqueeze(0))  # 近5
        q[q >= point_count] = point_count - 1
        dst_nodes = torch.cat((dst_nodes, q), dim=0)
    # for i in range(0,point_count*5):
    #     print(src_nodes[i])
    #     edge_attr[i,0] = torch.dist(src_nodes_3[int(src_nodes[i])],src_nodes_2[int(dst_nodes[i])])
    selected_values_2 = src_nodes_2[dst_nodes].float()
    selected_values_3 = src_nodes_3[src_nodes].float()
    edge_attr = torch.norm(selected_values_3 - selected_values_2, dim=1)
    edge_index = torch.cat((src_nodes.unsqueeze(0), dst_nodes.unsqueeze(0)), dim=0)
    return edge_index,edge_attr

def pygg(node_1,node_2,node_3,node_4,node_types):
    data = HeteroData().to(device)

    #建立节点
    # data['bp'].x = node_1
    data['ep'].x = node_2
    data['lp'].x = node_3
    data['cp'].x = node_4
    dict_attr={}
    aict_index = {}
    # #加入位置信息
    # src_nodes_4 = torch.nonzero(node_types == 4).squeeze()  # 类别 i 的节点
    # data['cp'].pos = src_nodes_4
    # src_nodes_3 = torch.nonzero(node_types == 3).squeeze()  # 类别 i 的节点
    # data['lp'].pos = src_nodes_3
    # src_nodes_2 = torch.nonzero(node_types == 2).squeeze()  # 类别 i 的节点
    # data['ep'].pos = src_nodes_2
    # src_nodes_1 = torch.nonzero(node_types == 1).squeeze()  # 类别 i 的节点
    # data['bp'].pos = src_nodes_1
    #添加边和和属性
    edge_index,edge_attr = get_edge(node_types, 4, 4)
    data['cp', 'ctc', 'cp'].edge_index = edge_index
    data['cp', 'ctc', 'cp'].edge_attr = edge_attr

    aict_index['cp', 'ctc', 'cp'] = edge_index
    dict_attr['cp', 'ctc', 'cp'] = edge_attr

    edge_index, edge_attr = get_edge(node_types, 3, 3)
    data['lp', 'ltl', 'lp'].edge_index = edge_index
    data['lp', 'ltl', 'lp'].edge_attr = edge_attr

    aict_index['lp', 'ltl', 'lp'] = edge_index
    dict_attr['lp', 'ltl', 'lp'] = edge_attr

    edge_index, edge_attr = get_edge(node_types, 2, 2)
    data['ep', 'ete', 'ep'].edge_index = edge_index
    data['ep', 'ete', 'ep'].edge_attr = edge_attr

    aict_index['ep', 'ete', 'ep'] = edge_index
    dict_attr['ep', 'ete', 'ep'] = edge_attr

    edge_index, edge_attr = get_edge(node_types, 3, 4)
    data['lp', 'ltc', 'cp'].edge_index = edge_index
    data['lp', 'ltc', 'cp'].edge_attr = edge_attr

    aict_index['lp', 'ltc', 'cp'] = edge_index
    dict_attr['lp', 'ltc', 'cp'] = edge_attr

    edge_index, edge_attr = get_edge(node_types, 3, 2)
    data['lp', 'lte', 'ep'].edge_index = edge_index
    data['lp', 'lte', 'ep'].edge_attr = edge_attr

    aict_index['lp', 'lte', 'ep'] = edge_index
    dict_attr['lp', 'lte', 'ep'] = edge_attr


    data = T.ToUndirected()(data) #无向图
    data = T.AddSelfLoops()(data) #自环图
    # data = T.Distance(cat=False)(data)


    return data,[aict_index,dict_attr]

def knn_search(data,query_point, k=5):
    # 计算查询点与数据集中所有点之间的距离
    data = data.float()
    query_point = query_point.float()
    distances = torch.cdist(query_point, data)

    # 找到前 k 个最近邻的索引
    _, indices = distances.topk(k, largest=False)
    return indices.squeeze(0)

# y = ygt()
# x =  torch.rand(1, 64, 128, 128)
# y(x)
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels, add_self_loops=False)
        self.lin2 = Linear(-1, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        return x

import numpy as np
def load_from_npy(npy_path):
    arrays = np.load(npy_path)
    print('have loaded all arrays from ', npy_path)
    return arrays
import Constants
import os

import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def graph_save():
    # input_line = load_from_npy(Constants.path_skel)  ####
    input_line = load_from_npy(Constants.path_skel_test)  ####第一维度存放分叉点，第二维度存放分割结果
    # x = load_from_npy(Constants.path_image_drive)
    input_line = torch.from_numpy(input_line).to(device)
    # x = torch.from_numpy(x).to(device)
    center_point = input_line[:, 0, :, :]#分叉点坐标
    b, c, H, W = [720,3,584,565]
    ting = input_line[:, 1, :, :]#分割图 (B,H,W)
    # types = torch.zeros([b,H, W]).to(device)
    del(input_line)
    for i in range(0,b):
        dict_edge = ()
        type_cp = center_point[i]
        # gin = x[i]  # 输入图像
        gin = torch.rand(c,H,W).to(device)
        tin = ting[i]
        types = get_type(tin)
        types[type_cp == 1] = 4
        node_1, node_2, node_3, node_4 = otof(gin, types, c, H)
        # 作图
        hetero,list1 = pygg(node_1, node_2, node_3, node_4, types)
        # for key, value in hetero.items():
        #     print("Key:", key, "Value:", value)
        with open(Constants.path_graph_edge_test+str(i)+'.pkl', 'wb') as f:
            pickle.dump(list1, f)
        print(i)
    # torch.save(types, Constants.path_graph_point_train)

def type_save():
    input_line = load_from_npy(Constants.path_skel)  ####第一维度存放分叉点，第二维度存放分割结果
    input_line = torch.from_numpy(input_line).to(device)
    center_point = input_line[:, 2, :, :]
    b, c, H, W = [720, 3, 512, 512]
    ting = input_line[:, 3, :, :]
    types = torch.zeros([b,H, W]).to(device)
    del (input_line)
    for i in range(0, b):
        type_cp = center_point[i]
        tin = ting[i]
        types[i] = get_type(tin)  # tcounts用来记录当前调的是哪个图的节点分类
        types[i][type_cp == 1] = 4
        # print(torch.sum(types[i]==1),torch.sum(types[i]==2),torch.sum(types[i]==3),torch.sum(types[i]==4))
        print(torch.sum(types[i]==1))
        print(torch.sum(types[i] == 2))
        print(torch.sum(types[i] == 3))
        print(torch.sum(types[i] == 4))
        print(i)
    # torch.save(types, Constants.path_graph_point_train)



import imageio
def graph_print():
    path = 'prob_1.png'
    input_line = load_from_npy(Constants.path_skel_test)  ####第一维度存放分叉点，第二维度存放分割结果
    ting = imageio.imread(path)
    input_line = torch.from_numpy(input_line).to(device)
    ting = torch.from_numpy(ting).to(device)
    center_point = input_line[:, 2, :, :]
    b, c, H, W = [1, 1, 584,565]

    for i in range(0, b):
        type_cp = center_point[i]
        tin = ting/255.
        types = get_type(tin)  # tcounts用来记录当前调的是哪个图的节点分类
        types[type_cp == 1] = 4
        # print(torch.sum(types[i]==1),torch.sum(types[i]==2),torch.sum(types[i]==3),torch.sum(types[i]==4))
        print(torch.sum(types==1))
        print(torch.sum(types == 2))
        print(torch.sum(types == 3))
        print(torch.sum(types == 4))
        print(i)

        new_color_retina = np.zeros(shape=(types.shape[0], types.shape[1], 3))
        # print(np.max(predicts_img), np.max(gt_img))
        for pixel_x in range(0, types.shape[0]):
            for pixel_y in range(0, types.shape[1]):
                if types[pixel_x, pixel_y] == 1:
                        new_color_retina[pixel_x, pixel_y, 1] = 0
                elif types[pixel_x, pixel_y] == 2:
                        new_color_retina[pixel_x, pixel_y, 2] = 255
                elif types[pixel_x, pixel_y] == 3:
                        new_color_retina[pixel_x, pixel_y, 1] = 255
                else:
                        new_color_retina[pixel_x, pixel_y, 1] = 255
        visualize(new_color_retina, 'color_map.png')
    # torch.save(types, Constants.path_graph_point_train)
# graph_save()
# type_save()
graph_print()