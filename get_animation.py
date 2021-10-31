import socket
import time
import pickle   
import cv2
import coordinates_extractor 
import math_transformation
import filtering
import music
import face_landmarks
import body_part
import sys
import numpy as np
import motion
import json
import params

def get_animation(file_name, recompute_flag=True):
    """
    В качестве аргумента идет имя картинки/ видео
    """

    # настройка сервера
    # address = ('127.0.0.1', 8052) 
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect(address)

    # считка аргумента
    input_file = file_name

    
    vidcap = cv2.VideoCapture(input_file)# считка файла
    success,image = vidcap.read()
    img_shape = np.array(image.shape[:2])# размер картинки
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))# определение фпс
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # количество фреймов

    if frame_count > 1:
        music_name = music.music_extractor(input_file) # достаем музыку

    
    if recompute_flag:
        
        coordinates3D =[]
        face_coords = []
        camera_modes = []
        emotions = []
        count = 0
        face_camera_mode = False
        R = []; T = []
        correct_left_elbow = []
        correct_right_elbow = []
        all_frames = []

        while success:
            
            all_frames.append(image)
            success,image = vidcap.read() # итерирование к следующему фрейму видео
            count += 1
        

        for batch in coordinates_extractor.dataloader(all_frames,params.batch_size):
            coords_2d,coords3d = coordinates_extractor.prediction(batch)
            coords3d = [math_transformation.knee_check(coord3d) for coord3d in coords3d]
            coords3d = [math_transformation.normalize_body(coord3d, math_transformation.unity_body_length, coord2d['ass'], img_shape) for coord3d, coord2d in zip(coords3d,coords_2d)]
            coordinates3D += coords3d



            for idx,image in enumerate(batch): # выделение 2д и 3д координат тела
                facebox = face_landmarks.face_detection(image)

                if facebox:

                    face_marks, marks = face_landmarks.face_landmarks(image,facebox)
                    face_camera_mode = body_part.face_mode(face_marks,image)
                    print(face_camera_mode)
                    face_coords.append(face_marks)

                    blink = motion.blink_detection(face_marks['left_eye'],face_marks['right_eye']) 
                    smile = motion.smile_detection(face_marks['lips'],face_marks['jaw'])
                    open_mouth = motion.mouth_aspect_detection(face_marks['lips'])
                    r,t = face_landmarks.face_vectors(image,marks)
                    R.append(r),T.append(t)

                else:
                    # нужно добавить проверку куда смотрит лицо, для поворота головы, как сделать хз. 
                    # а если смотрит прямо и предыдущих кадров нет, нужно передавать стандартыне координаты юнити
                    if len(face_coords) > 0:
                        face_marks = face_coords[-1]
                        r = R[-1]; t = T[-1]
                        R.append(r),T.append(t)

                        blink = motion.blink_detection(face_marks['left_eye'],face_marks['right_eye']) 
                        smile = motion.smile_detection(face_marks['lips'],face_marks['jaw'])
                        open_mouth = motion.mouth_aspect_detection(face_marks['lips'])
                        
                        
                    
                    else:
                        # face_marks = standart_face_coords
                        # r , t = standart coords
                        blink = False
                        smile = False
                        open_mouth = False
                
                emotions.append([blink,smile,open_mouth])
                

                if face_camera_mode:
                    camera_mode = 3
                    camera_modes.append(camera_mode)
                
                else:
                    camera_mode = body_part.camera_mode(image,coords_2d[idx])
                    camera_modes.append(camera_mode)

         
            

        coordinates3D = filtering.Kalman_preprocessing(coordinates3D) # сглаживание координат с помощью фильтра Калмана
        camera_mode = math_transformation.most_common(camera_modes)
        data = coordinates_extractor.json_prepare(coordinates3D,camera_mode,emotions,fps,R,T)
        return data
