from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os

import cv2
import numpy as np
import torch
import torch.utils.data
from model import create_model
from utils.image import get_affine_transform, transform_preds# for 2d
from utils.eval import  get_preds_3d,get_preds# for 2d
from models.msra_resnet import get_pose_net
# from params import *
import params


# Neural Network
model = get_pose_net(params.num_layers, params.heads)
checkpoint = torch.load(
  params.weight_path_3d, map_location=lambda storage, loc: storage)
state_dict = checkpoint['state_dict']
model.load_state_dict(state_dict, strict=False)
model = model.to(params.device)
model.eval()


def dict_create(skeleton_dict: dict, preds:np.array)->dict:
    
    coordinates = {}

    for label, coordinate in zip(skeleton_dict.values(),preds):

        coordinates[label] = coordinate
        
    return coordinates
    
    


def coords_create(image:np.array)->dict:
    # read image
    # image = cv2.imread(image_name)
    
    
    #image preprocessing
    s = max(image.shape[0], image.shape[1]) * 1.0
    c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
    trans_input = get_affine_transform(
      c, s, 0, [256, 256])
    inp = cv2.warpAffine(image, trans_input, (256, 256),
                         flags=cv2.INTER_LINEAR)
    inp = (inp / 255. - params.mean) / params.std
    inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
    inp = torch.from_numpy(inp).to(params.device)################################################
    #coordinates prediction
    out = model(inp)[-1]

    ## 2D coordinates
    pred_2d = get_preds(out['hm'].detach().cpu().numpy())[0]
    pred_2d = transform_preds(pred_2d, c, s, (64, 64))
    ## 3D coordinates
    pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), 
                         out['depth'].detach().cpu().numpy())[0]
    
    
    
    three_d_coords = dict_create(params.COCO_KEYPOINT_INDEXES,pred_3d)
    two_d_coords = dict_create(params.COCO_KEYPOINT_INDEXES,pred_2d)
    
    return three_d_coords,two_d_coords

def coords2d_to_3d(coords2d:dict,coords3d:dict)->dict:
    
    """
    add z coordinate to 2d coordinates
    """


    for key in coords3d.keys():
        
        depth = coords3d[key][2]
        coords2d[key] = np.append(coords2d[key],depth)
    
    return coords2d
    
def coordinates_update(coords:dict) -> dict:
    
    keys = coords.keys()
    
    for key in keys:
        
        if key == 'ass':
            continue
            
        if len(coords[key].shape) == 1:
            coords[key][:2] = coords['ass'][:2] - coords[key][:2]
            
        else:
            for i in range(len(coords[key])):
                coords[key][i][:2] = coords['ass'][:2] - coords[key][i][:2]

    coords['ass'] = np.array([0,0,0])
    return coords

def json_prepare(body_coords:list,camera_mode:int,emotions:list,fps:int,R:list,T:list):
    file = {}
    file['camera mode'] = camera_mode
    file['fps'] = fps
    file['data'] = {}
    
    for i,coords in enumerate(zip(body_coords,R,T,emotions)):
        data_frame = coords[0]
        data_frame['R'] = coords[1].tolist()
        data_frame['T'] = coords[2].tolist()
        data_frame['emotions'] = coords[3]
        file['data'][i] = data_frame
    return file


def dataloader(data:list,batch_size:int):
    
    for batch in range(int(len(data)/batch_size)):
        idx = batch * batch_size
        indexes = slice(idx,idx + batch_size)
        yield np.array(data[indexes])
        
def size_params(images:np.array):
    params = []
    
    for image in images:
        s = max(image.shape[0], image.shape[1]) * 1.0
        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        params.append([s,c])
    return params

def images_preprocessing(images:np.array):
    
    preprecessed_images = []
    scaled_params = []
    
    for image in (images):
        
        s = max(image.shape[0], image.shape[1]) * 1.0
        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        scaled_params.append([s,c])
        
        trans_input = get_affine_transform(c, s, 0, [256, 256])
        inp = cv2.warpAffine(image, trans_input, (256, 256),
                             flags=cv2.INTER_LINEAR)
#         print(inp.shape)
        inp = (inp / 255. - params.mean) / params.std
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        preprecessed_images.append(inp)
        
    preprecessed_images = np.stack(preprecessed_images).reshape(len(images),3,256,256)
    preprecessed_images = torch.from_numpy(preprecessed_images).to(params.device)
    
    return preprecessed_images,scaled_params

def prediction(images:np.array):
    
#     scale_params = size_params(images)
#     print(scale_params)
    images,scaled_params = images_preprocessing(images)
    
    out= model(images)[-1]
    xy = out['hm']
    depths = out['depth']
    
    coords_2d = []
    coords_3d = []
    
    for coords,depth,scale_param in zip(xy,depths,scaled_params):
        coords = coords.reshape(1,16,64,64)
        depth = depth.reshape(1,16,64,64)
        s = scale_param[0]
        c = scale_param[1]
        
        pred_2d = get_preds(coords.detach().cpu().numpy())
        pred_2d = pred_2d.reshape(16,2)
        pred_2d = transform_preds(pred_2d, c, s, (64, 64))
        pred_3d = get_preds_3d(coords.detach().cpu().numpy(), 
                         depth.detach().cpu().numpy())[0]

        three_d_coords = dict_create(params.COCO_KEYPOINT_INDEXES,pred_3d)
        two_d_coords = dict_create(params.COCO_KEYPOINT_INDEXES,pred_2d)
        
        
        coords_2d.append(two_d_coords)
        coords_3d.append(three_d_coords)
    return coords_2d,coords_3d