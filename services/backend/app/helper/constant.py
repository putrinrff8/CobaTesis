# === Area Wajah yang Diproses ===
COMPONENTS_SETUP = {
    'mulut': {
        'object_name': 'mouth',
        'object_rectangle': {"x_right": 67, "x_left": 48, "y_highest": 50, "y_lowest": 58},
        'pixel_shifting': {"pixel_x": 10, "pixel_y": 10},
        'object_dimension': {'width': 70, 'height': 35}
    },
    'alis': {
        'object_name': 'eyebrows',
        'object_rectangle': {"x_right": 26, "x_left": 17, "y_highest": 19, "y_lowest": 21},
        'pixel_shifting': {"pixel_x": 5, "pixel_y": 2},
        'object_dimension': {'width': 98, 'height': 14}
    }
}


BLOCKSIZE = 7
MODEL_PREDICTOR = "shape_predictor_68_face_landmarks.dat"
MODEL_SVM_EXTRACTION_FEATURE = "svm_model_rbf.joblib"
MODEL_SVM_EXTRACTION_FEATURE = "label_encoder.joblib"
#MODEL_SVM_4QMV = "svm_model_4qmv.joblib"
QUADRAN_DIMENSIONS = ['Q1', 'Q2', 'Q3', 'Q4']
FRAMES_DATA_QUADRAN_COMPONENTS = ['sumX', 'sumY', 'Tetha', 'Magnitude', 'JumlahQuadran']
