import dlib
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import torch
from torch_geometric.data import  Data
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def detect_landmarks(img):
   
    # set up the 68 point facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
   
    # detect faces in the image
    faces_in_image = detector(img, 0)
    
    # Process only first image
    face = faces_in_image[0]
    # assign the facial landmarks
    landmarks = predictor(img, face)

    # unpack the 68 landmark coordinates from the dlib object into a list
    landmarks_list = []
    master_node=[]
    for i in range(0, landmarks.num_parts):
        if(i>16):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))
        if(i==30):
            master_node = [landmarks.part(i).x, landmarks.part(i).y]
   
    return face, landmarks_list,master_node

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it  to the format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

def gaussian_noise(x,mu,std):
    noise = np.random.normal(mu, std, size = x.shape)
    x_noisy = x + noise
    return x_noisy 
# Check if a point is inside a rectangle
def rect_contains(rectangle, point):
    if point[0] < rectangle[0]:
        return False
    elif point[1] < rectangle[1]:
        return False
    elif point[0] > rectangle[2]:
        return False
    elif point[1] > rectangle[3]:
        return False
    return True

def in_list(c, classes):
    for i, sublist in enumerate(classes):
        # print(c,sublist)
        if (np.array_equal(np.array(c),np.array(sublist))):
            return i
    return -1
# Draw delaunay triangles
def draw_delaunay(img, subdiv,rect,master_node):
    print("draw")
    master_node[0] = master_node[0] - rect.left()
    master_node[1] = master_node[1] - rect.top()
    
    x, y, w, h = rect_to_bb(rect)
    triangle_list = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])
    node_feature= []
    idx=0
    for t in triangle_list:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        feature = []
        feature.append((pt1[0]-rect.left(),pt1[1]-rect.top()))
        if(in_list(feature,node_feature) == -1):
            node_feature.append((pt1[0]-rect.left(),pt1[1]-rect.top()))
    
        feature = []
        feature.append((pt2[0]-rect.left(),pt2[1]-rect.top()))
        if(in_list(feature,node_feature) == -1):
            node_feature.append((pt2[0]-rect.left(),pt2[1]-rect.top()))
       
        
        feature = []
        feature.append((pt3[0]-rect.left(),pt3[1]-rect.top()))
        if(in_list(feature,node_feature) == -1):
            node_feature.append((pt3[0]-rect.left(),pt3[1]-rect.top()))
        idx+=1
    edge_index= []
    edge_attr = []
    idx = 0
    idx_node= []
    for t in triangle_list:
        # if(idx==3):
        #     break
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        node1 = (t[0]-rect.left(), t[1]-rect.top())
        node2 = (t[2]-rect.left(), t[3]-rect.top())
        node3 = (t[4]-rect.left(), t[5]-rect.top())
        
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, (255,255,255), 1,  0)
            cv2.line(img, pt2, pt3, (255,255,255), 1,  0)
            cv2.line(img, pt3, pt1, (255,255,255), 1,  0)
            # pt1 -> pt2
            # print(pt1)
            node_idx_1 = in_list(node1,node_feature)
            node_idx_2 = in_list(node2,node_feature)
            node_idx_3 = in_list(node3,node_feature)
            idx_node+=[node_idx_1,node_idx_2,node_idx_3]

        idx+=1
    idx_node = np.unique(np.array(idx_node))
    node_feature = get_feature(node_feature,idx_node)
    print(len(node_feature))
    for t in triangle_list:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        node1 = (t[0]-rect.left(), t[1]-rect.top())
        node2 = (t[2]-rect.left(), t[3]-rect.top())
        node3 = (t[4]-rect.left(), t[5]-rect.top())
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            node_idx1 = in_list(node1,node_feature)
            node_idx2 = in_list(node2,node_feature)
            node_idx3 = in_list(node3,node_feature)
            dist_1 = np.absolute(np.linalg.norm(np.array(node_feature[node_idx1]) - np.array(node_feature[node_idx2]) ))
            dist_2 = np.absolute(np.linalg.norm(np.array(node_feature[node_idx2]) - np.array(node_feature[node_idx3]) ))
            dist_3 = np.absolute(np.linalg.norm(np.array(node_feature[node_idx3]) - np.array(node_feature[node_idx1]) ))
            edge_index.append([node_idx1,node_idx2])
            edge_index.append([node_idx2,node_idx3])
            edge_index.append([node_idx3,node_idx1])
            
            edge_attr.append([dist_1])
            edge_attr.append([dist_2])
            edge_attr.append([dist_3])
    idx_master = in_list(master_node,node_feature)
    for idx,feat in enumerate(node_feature):
            dist_1 = np.linalg.norm(np.array(node_feature[idx_master]) - np.array(node_feature[idx]) )
           
            edge_index.append([idx_master,idx])
            
            edge_attr.append([dist_1])
          
    # Scaling the node,edge 
    minMax = MinMaxScaler()
    edge_index = torch.tensor(edge_index)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    
    scaler =  preprocessing.StandardScaler()
    node_feature = scaler.fit_transform(node_feature)
    node_feature = np.asarray(node_feature)
    
    edge_attr = scaler.fit_transform(edge_attr)
    edge_attr = np.asarray(edge_attr)
    edge_attr = torch.tensor(edge_attr)

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
  
    
    return node_feature,edge_index,edge_attr,img
def delaunay_triangulation(img, points, rects,master_node):
    

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)
        
    # Draw delaunay triangles
    node_feature,edge_indices,edge_feats,img = draw_delaunay(img, subdiv,rects,master_node)
    return node_feature,edge_indices,edge_feats,img
def get_feature(node_feature,idx_list):
    feature = []
    for idx in idx_list:
        feature.append(node_feature[idx])
    return feature
    
def get_landmarks(image):
    # Compute and draw triangulation]
    rect, landmarks,master_node = detect_landmarks(image)
    node_feature,edge_indices,edge_feats,image = delaunay_triangulation(image, landmarks,rect,master_node)
    
    return node_feature,edge_indices,edge_feats,image
def save(data,idx,name,Train=True):
    print("Data: ",data)
    if(Train):
        # train_1 withoud rect,and -1,1
        path = "Dataset_Landmark/train"
    else:
        path = "Dataset_Landmark/test"
    torch.save(data, 
    os.path.join(path, 
                    f'data_{name}_{idx}.pt'))
import networkx as nx

# Convert it to graph data 
def to_data(node_feature,adj,attr,label,img):
    data = Data(x=node_feature, 
            edge_index=adj,
            edge_attr=attr,
            y=label
            ,image = img) 
    # print(data)
    # g = nx.DiGraph(directed=True)
    # g = torch_geometric.utils.to_networkx(data, to_undirected=False)
    # options = {
    # 'node_color': 'blue',
    # 'node_size': 100,
    # 'width': 3,
    # 'arrowstyle': '-|>',
    # 'arrowsize': 12,
    
    # }
   
    # # nx.draw(g)
    # nx.draw_networkx(g, arrows=True, **options)
    # plt.show()
    return data

def detect_face(image):
    detector = dlib.get_frontal_face_detector()
    detections = detector(image, 0)
    if(len(detections)<=0):
        return None
    return detections 

def adding_noise(nodes):
    mu = 0.0
    std = 0.05 * np.std(nodes)
    gaussian = gaussian_noise((nodes),mu,std)
    return gaussian

def adding_rotation(nodes,degree):
    len_nodes = len(nodes)
    theta = np.radians(degree)
    rotated_nodes = []
    c, s = np.cos(theta), np.sin(theta)
    for i in range(len_nodes):
        temp=[]

        rotated_x = float(float(nodes[i][0]) * c + float(nodes[i][1]) * s)
        rotated_y =float(-(float(nodes[i][0]) * s) + float(nodes[i][1]) * c)
        temp.append(rotated_x)
        temp.append(rotated_y)
        rotated_nodes.append(temp)
    rotated_nodes = np.asarray(rotated_nodes)
    rotated_nodes = torch.tensor(rotated_nodes, dtype=torch.float)
    return rotated_nodes
