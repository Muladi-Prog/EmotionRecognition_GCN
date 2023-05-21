import os
import cv2
import pandas as pd
import numpy as np
import imutils
import torch
import torchvision.transforms as transforms

# Import delaunay triangulation
from delaunay_triangulation import detect_face,get_landmarks,to_data,save,adding_noise,adding_rotation,rect_to_bb
# Transform image to tensor
transform_tensor = transforms.Compose([
    transforms.ToTensor()
])
# Transform image using gaussian blur
transform_blur = transforms.GaussianBlur(kernel_size=(7, 13), sigma=(0.1, 0.2))
# Normalization of img
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
transform_norm = transforms.Compose([
    transforms.Normalize(mean, std)
])


def splitting(label_list,image_list):
    d = {'label':label_list, 'value': image_list}
    df = pd.DataFrame(data=d)  # creating dataframe

    split_factor = 0.8
    #  initializing data frames
    train_data = pd.DataFrame(columns=['label', 'value'])
    test_data = pd.DataFrame(columns=['label', 'value'])

    print(df['label'].value_counts())

    # get all unique labels
    unique_labels = df['label'].unique()

    # store length of every unique label
    unique_lengths = {}
    for uni in unique_labels:
        unique_lengths[uni] = int(len(df[df.label == uni]) * split_factor)


    for uni in unique_labels:
        for _, row in df.iterrows():
            if(row['label'] == uni):
                if(unique_lengths[uni]):  # if unique length is not equal to 0
                    train_data = train_data.append({'label': row['label'], 'value': row['value']}, ignore_index=True)
                    unique_lengths[uni] = unique_lengths[uni] - 1  # minus unique lengths value
                else:

                    test_data = test_data.append({'label': row['label'], 'value': row['value']}, ignore_index=True)
    return train_data,test_data

# Get attribute of delaunay graph and save it
def get_attribute(idx,img,degree,path,flag,label,image):
    
    is_detected = detect_face(imutils.rotate(img, angle=degree))
    
    if(is_detected is None):
        print("Something went wrong! Face not detected!")
        return
    nodes,adj,attr,_ = get_landmarks(imutils.rotate(img, angle=degree)  )
    print("get_landmarks")
    nodes_ori = torch.tensor(nodes, dtype=torch.float)
    label = torch.tensor(np.asarray([label]), dtype=torch.int64)
    data = to_data(nodes_ori,adj,attr,label,image)
    save(data,idx,path,Train = flag)
    
def crop_resize(img,blur=False):
    detection = detect_face(img)
    if(detection is None):
        print("Something went wrong! Face not detected!")
        return None
    for rect in detection:
        x, y, w, h = rect_to_bb(rect)
        crop_image = img[y:y+h,x:x+w]
    resized_img = cv2.resize(crop_image,(227,227))
    img_tensor = transform_tensor(resized_img).unsqueeze(0)
    if(blur == True):
        img_tensor = transform_blur(img_tensor)
    img_tensor = transform_norm(img_tensor)
    return img_tensor

main_path = "FacialDataset/"
label_list = []
image_list = []
# Load facial dataset
for emotion in os.listdir(main_path):
    path_emotion = main_path + emotion +"/"
    for img in os.listdir(path_emotion):
        img_path = path_emotion + img
        im = cv2.imread(img_path)
        
        image_list.append(im)
        label_list.append(int(emotion)-1)

# Split data to train and test
train_data,test_data = splitting(label_list,image_list)

# Feature extraction part
for idx,(img,label) in enumerate( zip(train_data['value'],train_data['label']) ):
    
    print("Train: ",idx)
    # Flipped (img)
    flipped_image= cv2.flip(img, 1)
    
    # Gaussian (img)
    img_blur = crop_resize(img,True)
    # Rotate 5 degree(img)
    img_5 = imutils.rotate(img, angle=5)
    
    # Crop and resize the images
    img_5  =crop_resize(img_5)
    img_tensor = crop_resize(img)
    img_flip =crop_resize(flipped_image)



    # Get 
    get_attribute(idx,img,0,"ori",True,label,img_tensor)
    get_attribute(idx,img,5,"rotated_nodes_5",True,label,img_5) 
    get_attribute(idx,flipped_image,0,"ori_flipped",True,label,img_flip)

    # Gaussian noise
    is_detected = detect_face(img)
    
    if(is_detected is None):
        print("Something went wrong! Face not detected!")
        continue
    nodes,adj,attr,_ = get_landmarks(img  )
    
    label = torch.tensor(np.asarray([label]), dtype=torch.int64)
    
        # Adding Gaussian Noise
    gaussian_nodes = adding_noise(nodes)
    nodes_gaussian = torch.tensor(gaussian_nodes, dtype=torch.float)
    data = to_data(nodes_gaussian,adj,attr,label,img_blur)
    train_data = save(data,idx,"gaussian",Train = True)
    

for idx,(img,label) in enumerate( zip(test_data['value'],test_data['label']) ):
    print("Test: ",idx)
    img_tensor = crop_resize(img)
    print(img_tensor.size())
    get_attribute(idx,img,0,"ori",False,label,img_tensor)




