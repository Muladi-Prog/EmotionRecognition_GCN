import torch
import os

train_data =[]
test_data = []

def load_data():
    for idx in range(0,213):
        if(idx% 50 == 0):
            print("TRAIN IDX: ",idx)
        train_data.append(torch.load(os.path.join('train_last/', f'data_ori_{idx}.pt')))
        train_data.append(torch.load(os.path.join('train_last/', f'data_gaussian_{idx}.pt')))
        train_data.append(torch.load(os.path.join('train_last/', f'data_rotated_nodes_5_{idx}.pt')))
        # # image
        # train_data.append(torch.load(os.path.join('Dataset/train_normal_with_img/', f'image_ori_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_normal_with_img/', f'image_gaussian_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_normal_with_img/', f'image_rotated_nodes_5_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_normal_with_img/', f'image_ori_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_10_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_15_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_min5_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_min10_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_min15_{idx}.pt')))
        train_data.append(torch.load(os.path.join('train_last/', f'data_ori_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_5_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_10_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_15_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_min5_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_min10_flipped_{idx}.pt')))
        # train_data.append(torch.load(os.path.join('Dataset/train_1/', f'data_rotated_nodes_min15_flipped_{idx}.pt')))
        
    for idx in range(0,95):
        if(idx % 50 == 0):
            print("TEST IDX: ",idx)
        test_data.append(torch.load(os.path.join('test_last/', f'data_ori_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'data_gaussian_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'data_rotated_nodes_5_{idx}.pt')))
        # # image
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'image_ori_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'image_gaussian_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'image_rotated_nodes_5_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'image_ori_flipped_{idx}.pt')))
        
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_10_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_15_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_min5_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_min10_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_min15_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_normal_with_img/', f'data_ori_flipped_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_5_flipped_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_10_flipped_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_15_flipped_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_min5_flipped_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_min10_flipped_{idx}.pt')))
        # test_data.append(torch.load(os.path.join('Dataset/test_1/', f'data_rotated_nodes_min15_flipped_{idx}.pt')))
        
    return train_data,test_data
# from torch_geometric.data import DataLoader
# train_data,test_data = load_data()
# train_loader = DataLoader(test_data, batch_size=1, shuffle=True)
# # for i in train_loader:
# #     print(i)
# import numpy as np
# # print(len(test_data))
# # print(len(train_data))
# y = []
# for idx in range(len(test_data)):
#     print(test_data[idx])
#     y.append(np.array(test_data[idx].y[0]))
# y=np.array(y)
# print(np.unique(y))
# from sklearn.utils.class_weight import compute_class_weight
# class_weights = compute_class_weight(class_weight = 'balanced',classes= np.unique(y), y = y)
# class_weights=torch.tensor(class_weights,dtype=torch.float)
# print(class_weights)
