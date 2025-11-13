import os
import uuid
import time
import datetime
import dlib
import cv2
import joblib
import pandas as pd
import numpy as np
from flask import url_for
from werkzeug.utils import secure_filename
from app import response, app
from app.request.DataModel.DataTestStoreRequest import DataTestStoreRequest
from app.helper.preprocessing import get_frames_by_input_video, extract_component_by_images, draw_quiver_and_save_plotlib_image, convert_video_to_webm
from app.helper.helper import convert_video_to_avi, natural_sort_key, get_calculate_from_predict, convert_ndarray_to_list
from app.helper.poc import POC
from app.helper.vektor import Vektor
from app.helper.quadran import Quadran
from app.helper.constant import COMPONENTS_SETUP, FRAMES_DATA_QUADRAN_COMPONENTS, MODEL_PREDICTOR, MODEL_SVM_EXTRACTION_FEATURE , QUADRAN_DIMENSIONS, BLOCKSIZE

# ==============================================================================
# GLOBAL MODEL AND CONFIGURATION LOADING
# ==============================================================================
# Load all models once when the application starts to improve performance.
# This avoids reloading heavy models on every API request.
# ==============================================================================

try:
    # --- Dlib Model Loading ---
    DLIB_DETECTOR = dlib.get_frontal_face_detector()
    DLIB_PREDICTOR_PATH = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_MODEL'], MODEL_PREDICTOR)
    DLIB_PREDICTOR = dlib.shape_predictor(DLIB_PREDICTOR_PATH)
    print("[INFO] Dlib models loaded successfully.")

    # --- Base Paths ---
    BASE_MODEL_PATH = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_MODEL'])

    # --- Helper Function for Safe Loading ---
    def _load_joblib(filename):
        path = os.path.join(BASE_MODEL_PATH, filename)
        if not os.path.exists(path):
            print(f"[ERROR] Model file not found: {path}")
            return None
        return joblib.load(path)

    # --- Feature Info & PCA ---
    HYBRID_FEATURE_INFO = _load_joblib('hybrid_feature_selection_info.joblib')
    PCA_MODEL = _load_joblib('pca_100comp.joblib')

    # --- Scalers ---
    SCALERS = {
        'default': _load_joblib('scaler.joblib'),
        '4qmv': _load_joblib('4qmv_scaler.joblib'),
        'lda': _load_joblib('scaler_lda.joblib'),
        'hybrid': _load_joblib('hybrid_scaler.joblib'),
    }

    # --- Pre-trained Models & Encoders ---
    # Grouping models, scalers, and encoders together for easier management
    GLOBAL_MODELS = {
        "random_sampling": {
            "fitur_all_component": {
                "model": _load_joblib('svm_model_rbf.joblib'),
                "scaler": SCALERS['default'],
                "label_encoder": _load_joblib('label_encoder.joblib'),
                "except_features": ['Frame', 'Folder Path', 'Label']
            },
            "fitur_pca_component": {
                "model": _load_joblib('svm_model_rbf_pca.joblib'),
                "scaler": None, # PCA data is already scaled
                "label_encoder": _load_joblib('label_encoder_pca.joblib'),
                "except_features": []
            },
            "4qmv_all_component": {
                "model": _load_joblib('4qmv_svm_model_rbf.joblib'),
                "scaler": SCALERS['4qmv'],
                "label_encoder": _load_joblib('4qmv_label_encoder.joblib'),
                "except_features": ['Frame', 'Folder Path', 'Label']
            },
            "full_model_lda": {
                "model": _load_joblib('svm_full_model_lda.joblib'),
                "scaler": SCALERS['lda'],
                "label_encoder": _load_joblib('label_encoder_lda.joblib'),
                "except_features": ['Frame', 'Folder Path', 'Label']
            },
        }
    }

    # --- Hybrid Model Pre-processors (if any) ---
    HYBRID_PREFILTER = None
    HYBRID_RFE = None
    if HYBRID_FEATURE_INFO and HYBRID_FEATURE_INFO['method'] != 'direct_from_exploration':
        HYBRID_PREFILTER = _load_joblib('hybrid_prefilter.joblib')
        HYBRID_RFE = _load_joblib('hybrid_rfe.joblib')

    print("[INFO] All ML models, scalers, and encoders loaded globally.")

except Exception as e:
    print(f"[FATAL ERROR] Failed to load global models: {e}")
    # You might want to halt the application or handle this more gracefully
    DLIB_DETECTOR = None
    DLIB_PREDICTOR = None
    GLOBAL_MODELS = {}


# ==============================================================================
# API CONTROLLER
# ==============================================================================

def store():
    """
    Handles the main API request for video-based feature extraction and prediction.
    Orchestrates the pipeline: video processing, feature extraction,
    prediction, and response formatting.
    """
    print("[START] New prediction request received.")

    request_data = DataTestStoreRequest()
    if not request_data.validate():
        return response.error(422, 'Invalid request form validation', request_data.errors)

    file = request_data.file.data
    with_preview = request_data.with_preview.data
    new_filename_base = f'video-{str(uuid.uuid4())}'

    try:
        # --- 1. Video Processing ---
        video_paths, images_list, error = _handle_video_processing(file, new_filename_base)
        if error:
            return response.error(message=error)

        (
            file_path_video,
            file_path_output_images,
            new_filename_with_extension,
            file_path_video_response
        ) = video_paths

        # --- 2. Feature Extraction ---
        print(f"[INFO] Extracting features from frames in: {file_path_output_images}")
        df_fitur_all, df_quadran, preview_data_list = _extract_features_from_frames(
            file_path_output_images,
            images_list,
            with_preview
        )

        if df_fitur_all.empty or df_quadran.empty:
            return response.error(message="No faces detected or features extracted.")

        # --- 3. Save Features to CSV/Excel ---
        csv_urls = _save_feature_dataframes(df_fitur_all, df_quadran, new_filename_base)

        # --- 4. Prepare Feature Sets for Prediction ---
        print("[INFO] Preparing feature sets (PCA, Hybrid, etc.).")
        feature_sets = _prepare_feature_sets(
            df_fitur_all,
            df_quadran,
            except_columns=['Frame', 'Folder Path', 'Label']
        )

        # --- 5. Run All Predictions ---
        print("[INFO] Running predictions on all models.")
        predictions_result_all = _run_all_predictions(feature_sets)

        # --- 6. Format API Response ---
        print("[INFO] Formatting final API response.")
        response_data = _format_api_response(
            video_info=(file_path_video_response, new_filename_with_extension),
            csv_urls=csv_urls,
            predictions_result_all=predictions_result_all,
            preview_data_list=preview_data_list,
            with_preview=with_preview
        )

        print("[SUCCESS] Processing finished.\n")
        return response.success(200, 'Ok', response_data)

    except Exception as e:
        print(f"[ERROR] Unhandled exception in store(): {e}")
        import traceback
        traceback.print_exc()
        return response.error(500, message=f"An internal server error occurred: {e}")

# ==============================================================================
# HELPER FUNCTIONS (Private)
# ==============================================================================

def _handle_video_processing(file, new_filename_base):
    """
    Saves the uploaded video, converts it to AVI if needed,
    and extracts all frames using a helper.
    Returns paths and frame info.
    """
    file_extension = secure_filename(file.filename).split('.')[-1].lower()
    new_filename_with_extension = f"{new_filename_base}.{file_extension}"

    file_path_video = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], new_filename_with_extension)
    file_path_output_images = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_IMAGE'], 'output', new_filename_base)

    file.save(file_path_video)

    if file_extension != 'avi':
        print(f"[INFO] Converting .{file_extension} video to .avi...")
        converted_avi_filename = f"{new_filename_base}.avi"
        converted_avi_file_path = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_VIDEO'], converted_avi_filename)

        convert_video_to_avi(file_path_video, converted_avi_file_path)
        os.remove(file_path_video) # Clean up original

        file_path_video = converted_avi_file_path
        new_filename_with_extension = converted_avi_filename

    with app.app_context():
        file_path_video_response = url_for('static', filename=file_path_video.replace('\\', '/').replace('assets/', '', 1), _external=True)

    print(f"[INFO] Extracting frames (up to 200) from {new_filename_with_extension}...")
    images_list, error = get_frames_by_input_video(file_path_video, file_path_output_images, 200)
    if error:
        return None, None, error

    video_paths = (file_path_video, file_path_output_images, new_filename_with_extension, file_path_video_response)

    return video_paths, images_list, None


def _extract_features_from_frames(frames_dir, images_list, with_preview):
    """
    The core logic. Iterates through saved frames, detects faces,
    calculates POC, Vektor, and Quadran features.
    Separates preview data generation from feature calculation.
    """
    quadran_data_list = []
    all_features_data_list = []
    preview_data_list = []

    data_blocks_first_image = {component_name: None for component_name in COMPONENTS_SETUP}

    frame_counter = 0

    sorted_filenames = sorted(os.listdir(frames_dir), key=natural_sort_key)

    for filename in sorted_filenames:
        if not (filename.endswith(".jpg") or filename.endswith(".png")):
            continue

        image_path = os.path.join(frames_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Could not read image: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = DLIB_DETECTOR(gray)

        current_preview_data = {}
        if with_preview:
            current_preview_data = {
                "name": filename,
                "url": next((img['url'] for img in images_list if os.path.splitext(img['name'])[0] == os.path.splitext(filename)[0]), None),
                "components": {}
            }

        if frame_counter > 0:
            current_frame_all_features = {'Frame': f"{frame_counter + 1}({filename.split('.')[0]})"}
            current_frame_quadran_data = {'Frame': f"{frame_counter + 1}({filename.split('.')[0]})"}

        for rect in rects:
            shape = DLIB_PREDICTOR(gray, rect)

            for component_name, component_info in COMPONENTS_SETUP.items():

                sum_data_by_quadran = {
                    column: {quadrant: 0 for quadrant in QUADRAN_DIMENSIONS}
                    for column in FRAMES_DATA_QUADRAN_COMPONENTS
                }

                data_blocks_image_current, image_url = extract_component_by_images(
                    image=image,
                    shape=shape,
                    frameName=filename.split(".")[0],
                    objectName=component_info['object_name'],
                    objectRectangle=component_info['object_rectangle'],
                    pixelShifting=component_info['pixel_shifting'],
                    objectDimension=component_info['object_dimension'],
                    directoryOutputImage=frames_dir,
                    withPreview=with_preview
                )

                if with_preview:
                    current_preview_data["components"][component_name] = {"url_source": image_url}

                if data_blocks_first_image[component_name] is None:
                    data_blocks_first_image[component_name] = data_blocks_image_current
                    continue

                initPOC = POC(data_blocks_first_image[component_name], data_blocks_image_current, BLOCKSIZE)
                valPOC = initPOC.getPOC()

                initQuiv = Vektor(valPOC, BLOCKSIZE)
                quivData = initQuiv.getVektor()

                initQuadran = Quadran(quivData)
                quadran = initQuadran.getQuadran()

                if with_preview:
                    url_result = draw_quiver_and_save_plotlib_image(
                        dataBlockImage=data_blocks_image_current,
                        quivData=quivData,
                        frameName=filename.split(".")[0],
                        objectName=component_info['object_name'],
                        directoryOutputImage=frames_dir
                    )
                    current_preview_data["components"][component_name]["url_result"] = url_result

                # --- Feature Aggregation ---
                for i, quad in enumerate(quadran):
                    current_frame_all_features[f'{component_name}-X{i+1}'] = quad[1]
                    current_frame_all_features[f'{component_name}-Y{i+1}'] = quad[2]
                    current_frame_all_features[f'{component_name}-Tetha{i+1}'] = quad[3]
                    current_frame_all_features[f'{component_name}-Magnitude{i+1}'] = quad[4]

                    quadrant_label = quad[5]
                    if quadrant_label in QUADRAN_DIMENSIONS:
                        sum_data_by_quadran['sumX'][quadrant_label] += quad[1]
                        sum_data_by_quadran['sumY'][quadrant_label] += quad[2]
                        sum_data_by_quadran['Tetha'][quadrant_label] += quad[3]
                        sum_data_by_quadran['Magnitude'][quadrant_label] += quad[4]
                        sum_data_by_quadran['JumlahQuadran'][quadrant_label] += 1

                for quadran_dim in QUADRAN_DIMENSIONS:
                    for feature in FRAMES_DATA_QUADRAN_COMPONENTS:
                        column_name = f"{component_name}_{feature}_{quadran_dim}"
                        current_frame_quadran_data[column_name] = sum_data_by_quadran[feature][quadran_dim]

        if frame_counter > 0:
            current_frame_quadran_data['Folder Path'] = "data_test"
            current_frame_quadran_data['Label'] = "data_test"
            quadran_data_list.append(current_frame_quadran_data)

            current_frame_all_features['Folder Path'] = "data_test"
            current_frame_all_features['Label'] = "data_test"
            all_features_data_list.append(current_frame_all_features)

        if with_preview:
            preview_data_list.append(current_preview_data)

        frame_counter += 1

        # Stop processing if first image's blocks are still None (no faces found)
        if all(v is None for v in data_blocks_first_image.values()) and frame_counter > 10:
            print("[WARN] No faces detected in the first 10 frames. Aborting.")
            break

    df_fitur_all = pd.DataFrame(all_features_data_list)
    df_quadran = pd.DataFrame(quadran_data_list)

    return df_fitur_all, df_quadran, preview_data_list


def _save_feature_dataframes(df_fitur_all, df_quadran, new_filename_base):
    """
    Saves the extracted feature DataFrames to CSV and Excel files.
    Returns a dictionary of public URLs for the saved files.
    """
    output_csv_dir = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FOLDER_DATA'], new_filename_base)
    os.makedirs(output_csv_dir, exist_ok=True)

    # --- Save 'nilai-fitur-all-component' ---
    nilai_fitur_all_path_csv = os.path.join(output_csv_dir, 'nilai-fitur-all-component.csv')
    df_fitur_all.to_csv(nilai_fitur_all_path_csv, index=False, float_format=None)

    nilai_fitur_all_path_xlsx = os.path.join(output_csv_dir, 'nilai-fitur-all-component.xlsx')
    df_fitur_all.to_excel(nilai_fitur_all_path_xlsx, index=False, float_format=None)

    # --- Save '4qmv-all-component' ---
    nilai_4qmv_path_csv = os.path.join(output_csv_dir, '4qmv-all-component.csv')
    df_quadran.to_csv(nilai_4qmv_path_csv, index=False, float_format=None)

    nilai_4qmv_path_xlsx = os.path.join(output_csv_dir, '4qmv-all-component.xlsx')
    df_quadran.to_excel(nilai_4qmv_path_xlsx, index=False, float_format=None) # Note: Original code used .to_csv here, fixed to .to_excel

    # --- Generate Public URLs ---
    with app.app_context():
        csv_urls = {
            "nilai_fitur_asli_csv": url_for('static', filename=nilai_fitur_all_path_csv.replace('\\', '/').replace('assets/', '', 1), _external=True),
            "nilai_fitur_asli_xlsx": url_for('static', filename=nilai_fitur_all_path_xlsx.replace('\\', '/').replace('assets/', '', 1), _external=True),
            "nilai_4qmv_csv": url_for('static', filename=nilai_4qmv_path_csv.replace('\\', '/').replace('assets/', '', 1), _external=True),
            "nilai_4qmv_xlsx": url_for('static', filename=nilai_4qmv_path_xlsx.replace('\\', '/').replace('assets/', '', 1), _external=True),
        }
    return csv_urls


def _prepare_feature_sets(df_fitur_all, df_quadran, except_columns):
    """
    Applies preprocessing like PCA and Hybrid Feature Selection
    to the raw feature DataFrames.
    Returns a dictionary of prepared DataFrames ready for prediction.
    """
    feature_sets = {}

    # --- Base Features ---
    df_for_all = df_fitur_all.drop(columns=except_columns)
    df_for_4qmv = df_quadran.drop(columns=except_columns)

    feature_sets["fitur_all_component"] = df_for_all
    feature_sets["4qmv_all_component"] = df_for_4qmv

    # --- PCA Features ---
    if SCALERS['default'] and PCA_MODEL:
        df_scaled_pca = SCALERS['default'].transform(df_for_all)
        df_pca_transformed = PCA_MODEL.transform(df_scaled_pca)
        feature_sets["fitur_pca_component"] = pd.DataFrame(df_pca_transformed)

    # --- LDA Features ---
    # LDA model expects the 'fitur_all_component' data
    feature_sets["full_model_lda"] = df_for_all.copy()

    # --- Hybrid Features ---
    if HYBRID_FEATURE_INFO and SCALERS['hybrid']:
        df_scaled_hybrid = SCALERS['hybrid'].transform(df_for_all)

        if HYBRID_FEATURE_INFO['method'] == 'direct_from_exploration':
            selected_features = HYBRID_FEATURE_INFO['selected_features']
            df_hybrid = pd.DataFrame(df_scaled_hybrid, columns=df_for_all.columns)[selected_features]

        elif HYBRID_PREFILTER and HYBRID_RFE:
            df_prefiltered = HYBRID_PREFILTER.transform(df_scaled_hybrid)
            df_selected = HYBRID_RFE.transform(df_prefiltered)
            df_hybrid = pd.DataFrame(df_selected)

        else:
            print("[WARN] Hybrid feature preprocessors not found.")
            df_hybrid = pd.DataFrame() # Empty

        feature_sets["hybrid_component"] = df_hybrid

    return feature_sets


def _run_single_prediction(model, scaler, label_encoder, data):
    """
    Helper function to run prediction on a single model setup.
    Handles scaling, prediction, timing, and decoding.
    """
    try:
        if data.empty:
            return {"error": "Empty data frame."}

        # Apply scaling if a scaler is provided
        if scaler:
            data = scaler.transform(data)

        start_time = time.time()
        predictions = model.predict(data)
        end_time = time.time()

        elapsed_time = end_time - start_time
        decoded_predictions = label_encoder.inverse_transform(predictions)
        result_prediction, list_predictions = get_calculate_from_predict(decoded_predictions)

        return {
            "decoded_predictions": decoded_predictions,
            "result_prediction": result_prediction,
            "list_predictions": list_predictions,
            "testing_time_seconds": round(elapsed_time, 4)
        }
    except Exception as e:
        print(f"[ERROR] Prediction failed for model {model}: {e}")
        return {"error": str(e)}


def _run_all_predictions(feature_sets):
    """
    Iterates through the globally loaded models and runs predictions
    using the corresponding prepared feature sets.
    """
    predictions_result_all = {}

    for train_model_key, train_model_data in GLOBAL_MODELS.items():
        predictions_result_all[train_model_key] = {}

        for metode_key, config in train_model_data.items():

            if metode_key not in feature_sets:
                print(f"[WARN] Feature set '{metode_key}' not found. Skipping prediction.")
                continue

            data = feature_sets[metode_key]

            result = _run_single_prediction(
                model=config['model'],
                scaler=config['scaler'],
                label_encoder=config['label_encoder'],
                data=data
            )
            predictions_result_all[train_model_key][metode_key] = result

    return predictions_result_all


def _format_api_response(video_info, csv_urls, predictions_result_all, preview_data_list, with_preview):
    """
    Constructs the final JSON response dictionary from all processed data.
    """
    video_url, video_name = video_info

    response_data = {
        "video": {
            "url": video_url,
            "name": video_name,
        },
        "csv_file": {
            "nilai_fitur_asli_csv": csv_urls["nilai_fitur_asli_csv"],
            "nilai_fitur_asli_xlsx": csv_urls["nilai_fitur_asli_xlsx"],
            "nilai_4qmv_csv": csv_urls["nilai_4qmv_csv"],
            "nilai_4qmv_xlsx": csv_urls["nilai_4qmv_xlsx"],
        },
        "array_predictions": {},
        "result": {},
        "list_predictions": {},
        "testing_times": {}
    }

    # Unpack prediction results into the flat structure expected by the frontend
    for train_key, methods in predictions_result_all.items():
        for metode_key, result in methods.items():
            if "error" in result:
                continue

            key_name = f"{metode_key}with{train_key}"
            response_data["array_predictions"][key_name] = result['decoded_predictions'].tolist()
            response_data["result"][key_name] = result['result_prediction']
            response_data["list_predictions"][key_name] = result['list_predictions']
            response_data["testing_times"][key_name] = result['testing_time_seconds']

    # Merge predictions into the preview data if requested
    if with_preview:
        for i in range(len(preview_data_list)):
            if i == 0: # First frame has no prediction (no POC)
                preview_data_list[i]['prediction'] = None
            else:
                prediction_entry = {}
                for train_key, methods in predictions_result_all.items():
                    for metode_key, result in methods.items():
                        if "error" in result or (i-1) >= len(result['decoded_predictions']):
                            continue
                        key_name = f"{metode_key}with{train_key}"
                        prediction_entry[key_name] = result['decoded_predictions'][i-1]
                preview_data_list[i]['prediction'] = prediction_entry

        # Ensure data is JSON-serializable (handles numpy arrays)
        response_data["images"] = [
            item.tolist() if isinstance(item, np.ndarray) else item
            for item in preview_data_list
        ]

    return response_data
