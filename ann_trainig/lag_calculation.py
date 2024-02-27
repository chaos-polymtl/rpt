from main import * 

fix_lag = FixLag(number_of_points_training, number_of_detectors, x_0, y_0, z_0)
fix_lag.error_handshake()
