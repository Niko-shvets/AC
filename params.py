import numpy as np

left_side_bones = [['left_shoulder','left_elbow'],['left_elbow','left_wrist'],
                  ['left_hip','left_knee'],['left_knee','left_ankle']]

right_side_bones = [['right_shoulder','right_elbow'],['right_elbow','right_wrist'],
                  ['right_hip','right_knee'],['right_knee','right_ankle']]

# for unity_body_length calculation
hip = np.array([0,0,0])
spine = np.array([0, 0.1018159, 0.001315209])
spine1 = np.array([0, 0.1008345, -0.01000804])
spine2 = np.array([0, 0.09100011, -0.01373417])
neck = np.array([0, 0.1667167, -0.02516168])
left_up_leg = np.array([-0.08207782, -0.06751714, -0.01599556])
left_knee = np.array([0, -0.4437047, 0.002846426])
left_ankle = np.array([0, -0.4442787, -0.02982191])
left_shoulder = np.array([-0.1059237, -0.005245829, -0.0223212])
left_elbow = np.array([-0.2784152, 0, 0])
left_wrist = np.array([-0.2832884, 0, 0])
unity_ass_position = 4 # current camera position; it will be used to adjust body position at particular frame
unity_screen_shape = np.array([2, 3]) # TODO: create automation for this parameter


# # Banana man parameters
# hip = np.array([0, -0.0008799048, 0.05207037])
# spine = np.array([0, 0.006994551, 0])
# spine1 = np.array([0,0.007542891, 0])
# spine2 = np.array([0,0.007931685, 0])
# neck = np.array([0, 0.01014151, 0.000373998])
# left_up_leg = np.array([-0.004268162, -0.001836704, 0])
# left_knee = np.array([0, 0.01835469, 0])
# left_ankle = np.array([0, 0.02861734, 0])
# left_shoulder = np.array([0, 0.01007129, 0])
# left_elbow = np.array([0,0.01264537,0])
# left_wrist = np.array([0,0.01405164,0])
# unity_ass_position = 4
# unity_screen_shape = np.array([2, 3]) # TODO: create automation for this parameter


# for coordinate extractor file
mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

# Network parametrs
num_layers = 50
heads =  {'hm': 16, 'depth': 16}
device = 'cpu'
weight_path_3d = 'MODELS/fusion_3d_var.pth'
NUM_KPTS = 17
# motion params

EYE_AR_THRESH = 0.26
MOUTH_AR_THRESH = 0.6
SMILE_TRSH = 0.42

#skeleton params
edges = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], 
              [10, 11], [11, 12], [12, 8], [8, 13], [13, 14], [14, 15], 
              [6, 8], [8, 9]]

COCO_KEYPOINT_INDEXES = {
    0: 'right_ankle',
    1: 'right_knee',
    2: 'right_hip',
    3: 'left_hip',
    4: 'left_knee',
    5: 'left_ankle',
    6: 'ass',
    7: 'chest',
    8: 'neck',
    9: 'head',
    10: 'right_wrist',
    11: 'right_elbow',
    12: 'right_shoulder',
    13: 'left_shoulder',
    14: 'left_elbow',
    15: 'left_wrist',
}


high_boarder_lower_hor = 0.5
width_boarder_lower = 0.3

high_boarder_upper = 0.7
width_boarder_upper = 0.45

high_boarder_lower_ver = 0.3

body_part_region = {
    'left_shoulder':'left_arm',
    'left_elbow':'left_arm',
    'left_wrist':'left_arm',
    
    'right_shoulder':'right_arm',
    'right_elbow':'right_arm',
    'right_wrist':'right_arm',
    
    'left_hip':'left_leg',
    'left_knee':'left_leg',
    'left_ankle':'left_leg',
    
    'right_hip':'right_leg',
    'right_knee':'right_leg',
    'right_ankle':'right_leg',
    
    'ass' : 'ass',
    'chest': 'chest',
    'head':'head',
    'neck':'neck'}

body_part_seq = {
    'left_arm':   np.array(['left_shoulder','left_elbow','left_wrist']),
    'right_arm':   np.array(['right_shoulder','right_elbow','right_wrist']),
    'left_leg':   np.array(['left_hip','left_knee','left_ankle']),
    'right_leg':   np.array(['right_hip','right_knee','right_ankle']),
    'ass' : np.array(['ass']),
    'chest': np.array(['chest']),
    'head': np.array(['head']),
    'neck': np.array(['neck'])
}

left_arm = ['left_shoulder','left_elbow','left_wrist']
right_arm = ['right_shoulder','right_elbow','right_wrist']

left_leg = ['left_hip','left_knee','left_ankle']
right_leg = ['right_hip','right_knee','right_ankle']
bottom_limbs = ['right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle']

left_side_bottom_bones = [['ass','left_hip'],['left_hip','left_knee'],['left_knee','left_ankle']]

right_side_bottom_bones = [['ass','right_hip'],['right_hip','right_knee'],['right_knee','right_ankle']]
batch_size = 20