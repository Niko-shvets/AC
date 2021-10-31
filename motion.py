from scipy.spatial import distance as dist
import numpy as np
import params

def eye_aspect_ratio(eye:np.array)->float:
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def blink_detection(left_eye:np.array,right_eye:np.array):
    blink = False

    left_eye_ratio = eye_aspect_ratio(left_eye)
    right_eye_ratio = eye_aspect_ratio(right_eye)

    mean_ration = (left_eye_ratio + right_eye_ratio)/2

    if mean_ration < params.EYE_AR_THRESH:
        blink = True
    
    return blink

def mouth_aspect_detection(mouth:np.array)->float:

    open_mouth = False
    # compute the euclidean distances between the two sets of
    # vertical mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
    B = dist.euclidean(mouth[4], mouth[8]) # 53, 57

    # compute the euclidean distance between the horizontal
    # mouth landmark (x, y)-coordinates
    C = dist.euclidean(mouth[0], mouth[6]) # 49, 55

    # compute the mouth aspect ratio
    mar = (A + B) / (2.0 * C)

    if mar > params.MOUTH_AR_THRESH:
        open_mouth = True
    # return the mouth aspect ratio
    return open_mouth


def smile_detection(mouth:np.array,jaw:np.array)->float:
    smile = False

    lips_width = abs(mouth[0][0] - mouth[6][0])
    jaw_width = abs(jaw[0][0] - jaw[-1][0])
    smile_ratio = lips_width/jaw_width
    
    if smile_ratio > params.SMILE_TRSH:
        smile = True
    return smile