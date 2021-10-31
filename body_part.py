import numpy as np
import math
import params
import math_transformation
import cv2
import face_landmarks

def corner_neighbors(corner_parts:list):
    
    neighbor_parts = []
    
    for part in corner_parts:
        body_region = params.body_part_region[part]
        body_seq = params.body_part_seq[body_region]
        body_part_index = np.where(body_seq == part)[0][0]
        
        try:
            neighbor = body_seq[body_part_index - 1]
        except:
            continue
        else:
            neighbor_parts.append(part)
           
        
    if len(neighbor_parts) > 0:
        neighbor_marker = True
    
    else:
        neighbor_marker = False
    
    return neighbor_marker

def camera_mode(image:np.array,body_coords:dict):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    y = int(image.shape[0] * 0.1)
    x = int(image.shape[1] * 0.1)
    
    corner_parts = []
    
    for key in body_coords:
        if body_coords[key][0] > x and body_coords[key][1] < y:
            corner_parts.append(key)
            
    
    if len(corner_parts) > 1:
        neighbor_marker = corner_neighbors(corner_parts)
        if neighbor_marker:
            
            camera_mode = 2
#             return camera_mode
        elif 'ass' in corner_parts:
            camera_mode = 2
        
        else:
            key_limb_marker = False
            for key_limb in params.bottom_limbs:
                if key_limb in corner_parts:
                    key_limb_marker = True
                    break
            if key_limb_marker:
                marker = math_transformation.check_sides(body_coords,100,params.left_side_bottom_bones,params.right_side_bottom_bones)
                if marker:
                    camera_mode = 2
                else:
                    camera_mode = 1
    else:
        camera_mode = 1
        
    return camera_mode

def face_mode(face_landmark,image):

    img_shape = np.array(image.shape)
    
    face_high,face_width = face_landmarks.face_size(face_landmark,image)   

    if img_shape[0] > img_shape[1]:
        high_boarder_lower = params.high_boarder_lower_ver
        
    else:
        high_boarder_lower = params.high_boarder_lower_hor

    if face_high > high_boarder_lower:

        face_mode = True
    
    else:
        face_mode = False
        

 
    return face_mode



