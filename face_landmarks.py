import dlib
import cv2
import numpy as np
from head_pose_estimation import PoseEstimator

dlib_model_path = 'head_pose_estimation/assets/shape_predictor_68_face_landmarks.dat'
shape_predictor = dlib.shape_predictor(dlib_model_path)
face_detector = dlib.get_frontal_face_detector()



def shape_to_np(shape):
    coords = np.zeros((68, 2))
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def get_face(detector, image, cpu=True):
    if cpu:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        try:
            box = detector(image)[0]
            x1 = box.left()
            y1 = box.top()
            x2 = box.right()
            y2 = box.bottom()
            return [x1, y1, x2, y2]
        except:
            return None
    else:
        image = cv2.resize(image, None, fx=0.5, fy=0.5)
        box = detector.detect_from_image(image)[0]
        if box is None:
            return None
        return (2*box[:4]).astype(int)

def face_detection(image:np.array):
    facebox = get_face(face_detector, image, 'cpu')
    return facebox


def face_landmarks(image: np.array,facebox:list):
   
    landmarks = {}
    
    
    face = dlib.rectangle(left=facebox[0], top=facebox[1], 
                                      right=facebox[2], bottom=facebox[3])
    marks = shape_to_np(shape_predictor(image, face))
    
    landmarks['lips'] = marks[48:]
    landmarks['right_eyebrow'] = marks[17:22]
    landmarks['left_eyebrow'] = marks[22:27]
    landmarks['nose'] = marks[27:36]
    landmarks['left_eye'] = marks[42:48]
    landmarks['right_eye'] = marks[36:42]
    landmarks['jaw'] = marks[3:15]
    return landmarks,marks



    
def face_size(face_land, image):
    height = image.shape[0]
    width = image.shape[1]

    min_lips_y = max(face_land['lips'][:,1])
    max_right_eyebrow_y = min(face_land['right_eyebrow'][:,1])
    max_left_eyebrow_y = min(face_land['left_eyebrow'][:,1])
    max_eybrow_y = min([max_right_eyebrow_y,max_left_eyebrow_y])

    min_left_eyebrow_x = min(face_land['left_eyebrow'][:,0])
    max_right_eyebrow_x = max(face_land['right_eyebrow'][:,0])

    face_high = (abs(min_lips_y - max_eybrow_y)*2.3) /height
    face_width = (abs(max_right_eyebrow_x - min_left_eyebrow_x)*1.5)/width
    return face_high,face_width


def threed_face(face,body):
    
    head_depth = body['head'][2]
    
    for key in face.keys():
        
        if len(face[key].shape) == 1:
            face[key] = np.append(face[key],head_depth)

        else:
            depth_column = [[head_depth]] * len(face[key])
            face[key] = np.append(face[key], depth_column, axis=1)

           
                
    return face

def face_vectors(image:np.array,marks:np.array):
   
    pose_estimator = PoseEstimator(img_size=image.shape[:2])
    error, R, T = pose_estimator.solve_pose_by_68_points(marks)
    # print(R)
    # print(T)
    return R,T#np.expand_dims(R,0),np.expand_dims(T,0)