import numpy as np
import math
from pykalman import KalmanFilter








def Kalman1D(observations,damping=1):
    # To return the smoothed time series data
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state



def Kalman_preprocessing(coords:list):
    
    length_measurments = len(coords)
    
    measurements_x = {}
    measurements_y = {}
    measurements_z = {}
    
    keys = coords[0].keys()
    
    for key in keys:
        measurements_x[key] = Kalman1D([cordinate[key][0] for cordinate in coords])
        measurements_y[key] = Kalman1D([cordinate[key][1] for cordinate in coords])
        measurements_z[key] = Kalman1D([cordinate[key][2] for cordinate in coords])
    
    filtered_coords = []
    
    for i in range(length_measurments):
        
        current_dict = {}
        
        for key in keys:
            current_dict[key] = list(np.round(np.array([measurements_x[key][i],measurements_y[key][i],measurements_z[key][i]]).reshape(-1), 3))
        
        filtered_coords.append(current_dict)
    
    
    return filtered_coords


