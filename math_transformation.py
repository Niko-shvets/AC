import numpy as np
import math
import params



unity_body_length = np.linalg.norm(params.hip + params.spine + params.spine1 + params.spine2 + params.neck)
unity_up_leg_length = np.linalg.norm(params.left_knee)
unity_lower_leg_length = np.linalg.norm(params.left_ankle)
unity_arm_length = np.linalg.norm(params.left_elbow)
unity_forearm_length = np.linalg.norm(params.left_wrist)



def point_angle(p1:np.array, p2:np.array)->float:
    """
    calcualte angle between 2 points
    """
    
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])

    return math.degrees((ang1 - ang2) % (2 * np.pi))


def get_angle(a:np.array,b:np.array,c:np.array)->float:

    """
    calculate angle between 3 points via vectors
    """
    
    ba = a - b
    cb = b - c
    
    ba_angle = math.atan2(ba[0], ba[1])
    cb_anlge = math.atan2(cb[0], cb[1])
    
    cba_angle = cb_anlge - ba_angle
    cba_angle_deg = math.degrees(cba_angle)
    
    return cba_angle_deg

def human_angles(coords:dict)->dict:
    
    angles = {}
    
    angles['left_shoulder'] = get_angle(coords['left_elbow'],coords['left_shoulder'],coords['left_hip']) - 90
    angles['right_shoulder'] = get_angle(coords['right_elbow'],coords['right_shoulder'],coords['right_hip']) + 90
    
    angles['left_elbow'] = get_angle(coords['left_wrist'],coords['left_elbow'],coords['left_shoulder']) 
    angles['right_elbow'] = get_angle(coords['right_wrist'],coords['right_elbow'],coords['right_shoulder'])
    
        
    angles['left_hip_body'] = get_angle(coords['left_knee'],coords['left_hip'],coords['left_shoulder'])
    angles['right_hip_body'] = get_angle(coords['right_knee'],coords['right_hip'],coords['right_shoulder'])
        
    
    angles['left_knee'] = get_angle(coords['left_ankle'],coords['left_knee'],coords['left_hip'])
    angles['right_knee'] = get_angle(coords['right_ankle'],coords['right_knee'],coords['right_hip'])
    
    angles['head'] = point_angle(coords['head'][:2],coords['neck'][:2])
        

    return angles


def get_forward_vec(a:np.array,b:np.array,c:np.array)->np.array:
    """
    Returns normalized vector of a plane given by 3 points
    """
    v1 = b - a
    v2 = c - a
    
    forward_vec = np.cross(v2, v1)
    forward_vecn = forward_vec / np.linalg.norm(forward_vec)
    
    return forward_vecn

def get_angles_oxoy(vec:np.array)->float:
    # initial position of a vector is [0,0,1]. It leads to additional rotation by +- 90 degrees
    
    angle_ox = math.degrees(math.atan2(vec[2], vec[1])) - 90    
    angle_oy = math.degrees(math.atan2(vec[2], vec[0])) + 90
    
    return angle_ox, angle_oy

def bone_length(bone1: np.array,bone2: np.array)->float:
    """
    Calculate length between 2 points
    """
    
    x1,y1 = bone1[0],bone1[1]
    x2,y2 = bone2[0],bone2[1]
    
    dist = np.sqrt((x2-x1)**2 + (y2 - y1)**2)
    
    return dist

def check_sides(coords:dict,trsh:int,left_side,right_side)->bool:

    """
    This function compare length of left and right of bones paar: left hand and right hand
    This function allow to understand how correct is  detected skeleton 
    return bool value
    """
    
    count = 0
    disbalance = False
    
    
    for left, right in zip(left_side,params.right_side):
        
        left_bones_dist = bone_length(coords[left[0]],coords[left[1]])
        right_bones_dist = bone_length(coords[right[0]],coords[right[1]])
        
        length_difference = abs(right_bones_dist - left_bones_dist) 
        
        if length_difference > trsh:
            count += 1
        
    if count >= 2:
        
        disbalance = True
        
  
        
    return disbalance


def normalize_body(body_dict:dict, 
                   unity_body_length:float, 
                   ass_2d:np.array, 
                   img_shape:np.array)->dict:
    """
    Returns normalized body skeleton. Normalization is in accordance to Unity model
    """
    
    helper_arr = np.array([1,1,1])

    image_body_length = np.linalg.norm((body_dict['ass'] - body_dict['neck']))
    scale_factor_body = unity_body_length / image_body_length

    # new_body_dict = {'ass': np.array([0,0,0], dtype=np.float32)}
    new_body_dict = {'ass': get_ass_location(image_body_length, unity_body_length, params.unity_ass_position, ass_2d, img_shape)}
    new_body_dict['neck'] = (body_dict['ass'] - body_dict['neck']) * scale_factor_body * helper_arr
    new_body_dict['chest'] = (body_dict['ass'] - body_dict['chest']) * scale_factor_body * helper_arr
    
    new_body_dict['right_hip'] = (body_dict['ass'] - body_dict['right_hip']) * scale_factor_body * helper_arr
    new_body_dict['right_knee'] = (body_dict['right_hip'] - body_dict['right_knee']) * helper_arr
    new_body_dict['right_knee'] = new_body_dict['right_knee'] * unity_up_leg_length / np.linalg.norm(new_body_dict['right_knee'])
    new_body_dict['right_ankle'] = (body_dict['right_knee'] - body_dict['right_ankle']) * helper_arr
    new_body_dict['right_ankle'] = new_body_dict['right_ankle'] * unity_lower_leg_length / np.linalg.norm(new_body_dict['right_ankle'])

    new_body_dict['left_hip'] = (body_dict['ass'] - body_dict['left_hip']) * scale_factor_body * helper_arr
    new_body_dict['left_knee'] = (body_dict['left_hip'] - body_dict['left_knee']) * helper_arr
    new_body_dict['left_knee'] = new_body_dict['left_knee'] * unity_up_leg_length / np.linalg.norm(new_body_dict['left_knee'])
    new_body_dict['left_ankle'] = (body_dict['left_knee'] - body_dict['left_ankle']) * helper_arr
    new_body_dict['left_ankle'] = new_body_dict['left_ankle'] * unity_lower_leg_length / np.linalg.norm(new_body_dict['left_ankle'])
    
    new_body_dict['right_shoulder'] = (body_dict['chest'] - body_dict['right_shoulder']) * scale_factor_body * helper_arr
    new_body_dict['right_elbow'] = (body_dict['right_shoulder'] - body_dict['right_elbow']) * helper_arr
    new_body_dict['right_elbow'] = new_body_dict['right_elbow'] * unity_arm_length / np.linalg.norm(new_body_dict['right_elbow'])
    new_body_dict['right_wrist'] = (body_dict['right_elbow'] - body_dict['right_wrist']) * helper_arr
    new_body_dict['right_wrist'] = new_body_dict['right_wrist'] * unity_forearm_length / np.linalg.norm(new_body_dict['right_wrist'])
    
    new_body_dict['left_shoulder'] = (body_dict['chest'] - body_dict['left_shoulder']) * scale_factor_body * helper_arr
    new_body_dict['left_elbow'] = (body_dict['left_shoulder'] - body_dict['left_elbow']) * helper_arr
    new_body_dict['left_elbow'] = new_body_dict['left_elbow'] * unity_arm_length / np.linalg.norm(new_body_dict['left_elbow'])
    new_body_dict['left_wrist'] = (body_dict['left_elbow'] - body_dict['left_wrist']) * helper_arr
    new_body_dict['left_wrist'] = new_body_dict['left_wrist'] * unity_forearm_length / np.linalg.norm(new_body_dict['left_wrist'])

    

    return new_body_dict



def knee_check(coords:dict)->dict:
    """
    Corrects knee rotation. Z axis goes into the screen
    """
    scale_knee = 1
    flag_knee = False
    # TODO: create flag for theses conditions to check if it works properly
    if (coords['left_knee'][2] > coords['left_ankle'][2]): #and (coords['left_knee'][2] > coords['left_hip'][2]):
        new_depth = coords['left_ankle'][2] - (coords['left_knee'][2] - coords['left_ankle'][2])
        coords['left_knee'][2] = new_depth / scale_knee
        flag_knee = True
        # print("Incorrect left knee rotation")
    
    if (coords['right_knee'][2] > coords['right_ankle'][2]): #and (coords['right_knee'][2] > coords['right_hip'][2]):
        new_depth = coords['right_ankle'][2] - (coords['right_knee'][2] - coords['right_ankle'][2])
        coords['right_knee'][2] = new_depth / scale_knee
        flag_knee = True
        # print("Incorrect rigth knee rotation")

    # if flag_knee:
        # print("Incorrect knee rotation")

    return coords


def knee_check2(coords:dict)->dict:
    """
    Corrects knee rotation. Z axis goes into the screen.
    Version 2 of the knee check algorithm. It is based on the angles between bones
    """

    flag_knee = False

    left_knee_angle = point_angle(coords['left_hip'][[1,2]], coords['left_ankle'][[1,2]])
    right_knee_angle = point_angle(coords['right_hip'][[1,2]], coords['right_ankle'][[1,2]])

    if left_knee_angle > 0:
        coords['left_knee'][2] = coords['left_ankle'][2]
        flag_knee = True

    if right_knee_angle > 0:
        coords['right_knee'][2] = coords['right_ankle'][2]
        flag_knee = True

    if flag_knee:
        print("Incorrect knee rotation")

    return coords



def get_ass_location(image_body_length:float, 
                     unity_body_length:float, 
                     unity_ass_position:float, 
                     ass_2d:np.array,
                     img_shape:np.array )->np.array:

    z_coord = (unity_ass_position * image_body_length) / unity_body_length  - unity_ass_position
    new_z_coord = - z_coord / 1.5

    ass_2d -= img_shape/2
    ass_2d *= -1
    new_ass_2d = ass_2d * params.unity_screen_shape / img_shape
    # x,y = new_ass_2d
    # new_ass_2d -= unity_screen_shape/2
    # new_ass_2d *= -1
    
    new_ass_2d[1] += 1 # add 1 as camera location

    return np.array([new_ass_2d[0], new_ass_2d[1], new_z_coord], dtype=np.float32)





def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

def most_common(lst):
    return max(set(lst), key=lst.count)


def elbow_angle_check(coords:dict):
    
    left_elbow_angle = get_angle(coords['left_shoulder'], coords['left_elbow'],coords['left_wrist'])
    right_elbow_angle = get_angle(coords['right_shoulder'], coords['right_elbow'],coords['right_wrist'])

    left_elbow_falg = False
    right_elbow_falg = False

    if left_elbow_angle < 0:
        print('-----'*10)
        print('incorrect left elbow')
        left_elbow_falg = True
    if right_elbow_angle < 0:
        print('-----'*10)
        print('incorrect right elbow')
        right_elbow_falg = True
    return left_elbow_falg,right_elbow_falg


def elbow_angle_check_2(coords:dict):
    
    left_elbow_angle = get_angle(coords['left_shoulder'], coords['left_elbow'],coords['left_wrist'])
    right_elbow_angle = get_angle(coords['right_shoulder'], coords['right_elbow'],coords['right_wrist'])

    left_elbow_falg = False
    right_elbow_falg = False

    if left_elbow_angle < 0:
        print('-----'*10)
        print('incorrect left elbow')
        new_depth = coords['left_wrist'][2] - (coords['left_elbow'][2] - coords['left_wrist'][2])
        coords['left_elbow'][2] = new_depth
        left_elbow_falg = True

    if right_elbow_angle < 0:

        new_depth = coords['right_wrist'][2] - (coords['right_elbow'][2] - coords['right_wrist'][2])
        coords['right_elbow'][2] = new_depth
        left_elbow_falg = True
        print('-----'*10)
        print('incorrect right elbow')
        right_elbow_falg = True
    return coords