import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = 'best_random_forest_model.pkl'
SCALER_PATH = 'scaler.pkl'  # Si usaste escalado

_model = None
_scaler = None

# Mapeos de variables categóricas (label encoding)
SEX_MAP = {'F': 0, 'M': 1}
CHEST_PAIN_MAP = {'ASY': 0, 'ATA': 1, 'NAP': 2, 'TA': 3}
RESTING_ECG_MAP = {'LVH': 0, 'Normal': 1, 'ST': 2}
EXERCISE_ANGINA_MAP = {'N': 0, 'Y': 1}
ST_SLOPE_MAP = {'Down': 0, 'Flat': 1, 'Up': 2}

def load_model_and_scaler():
    """Carga el modelo y el scaler (si existe)."""
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
        print("Modelo cargado exitosamente.")

        if os.path.exists(SCALER_PATH):
            _scaler = joblib.load(SCALER_PATH)
            print("Scaler cargado exitosamente.")
        else:
            print("Advertencia: No se encontró scaler. Se asume datos sin escalar.")

def predict(features_list):
    """
    Realiza una predicción.
    Args:
        features_list: Lista de 11 características en el orden:
        [Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
         RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope]
    Returns:
        Tuple: (predicción (int), probabilidad (float) o mensaje de error)
    """
    load_model_and_scaler()
    if _model is None:
        return None, "Modelo no cargado."

    try:
        # --- APLICAR MAPEOS A LAS VARIABLES CATEGÓRICAS ---
        # Índices según el orden de la lista:
        # 0: Age (numérico)
        # 1: Sex (categórico)
        # 2: ChestPainType (categórico)
        # 3: RestingBP (numérico)
        # 4: Cholesterol (numérico)
        # 5: FastingBS (numérico, ya es 0/1)
        # 6: RestingECG (categórico)
        # 7: MaxHR (numérico)
        # 8: ExerciseAngina (categórico)
        # 9: Oldpeak (numérico)
        # 10: ST_Slope (categórico)

        # Convertir a números las categóricas usando los mapeos
        features_list[1] = SEX_MAP[features_list[1]]                 # Sex
        features_list[2] = CHEST_PAIN_MAP[features_list[2]]         # ChestPainType
        features_list[6] = RESTING_ECG_MAP[features_list[6]]        # RestingECG
        features_list[8] = EXERCISE_ANGINA_MAP[features_list[8]]    # ExerciseAngina
        features_list[10] = ST_SLOPE_MAP[features_list[10]]         # ST_Slope

        # Asegurar que los numéricos sean float
        features_list[0] = float(features_list[0])   # Age
        features_list[3] = float(features_list[3])   # RestingBP
        features_list[4] = float(features_list[4])   # Cholesterol
        features_list[5] = float(features_list[5])   # FastingBS (ya es 0/1, pero lo dejamos como float)
        features_list[7] = float(features_list[7])   # MaxHR
        features_list[9] = float(features_list[9])   # Oldpeak

        # Convertir a numpy array y dar forma
        features_array = np.array(features_list).reshape(1, -1)

        # Aplicar escalado si existe
        if _scaler:
            features_array = _scaler.transform(features_array)

        # Predecir
        prediction = _model.predict(features_array)[0]
        probability = None
        if hasattr(_model, "predict_proba"):
            proba = _model.predict_proba(features_array)[0]
            probability = float(proba[1])  # Probabilidad de clase positiva (HeartDisease=1)

        return int(prediction), probability

    except Exception as e:
        return None, f"Error en la predicción: {str(e)}"