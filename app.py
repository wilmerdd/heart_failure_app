from flask import Flask, render_template, request, jsonify
import sqlite3
import datetime
import os
from model_utils import predict, load_model_and_scaler

app = Flask(__name__)

# --- Configuración de la Base de Datos (SQLite para historial) ---
DATABASE = 'database.db'

def init_db():
    """Inicializa la base de datos creando la tabla si no existe."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            Age INTEGER, Sex TEXT, ChestPainType TEXT, RestingBP INTEGER,
            Cholesterol INTEGER, FastingBS INTEGER, RestingECG TEXT,
            MaxHR INTEGER, ExerciseAngina TEXT, Oldpeak REAL, ST_Slope TEXT,
            HeartDisease INTEGER,
            Probability REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Base de datos inicializada.")

def save_prediction_to_db(data, prediction, probability):
    """Guarda una predicción en la base de datos."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO predictions (
            timestamp, Age, Sex, ChestPainType, RestingBP, Cholesterol,
            FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope,
            HeartDisease, Probability
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        timestamp, data['Age'], data['Sex'], data['ChestPainType'], data['RestingBP'],
        data['Cholesterol'], data['FastingBS'], data['RestingECG'], data['MaxHR'],
        data['ExerciseAngina'], data['Oldpeak'], data['ST_Slope'],
        prediction, probability
    ))
    conn.commit()
    conn.close()

def get_prediction_history(limit=50):
    """Obtiene el historial de predicciones."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, Age, Sex, ChestPainType, RestingBP, Cholesterol,
               FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope,
               HeartDisease, Probability
        FROM predictions ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

# --- Inicialización al arrancar la app ---
init_db()
try:
    load_model_and_scaler()
except FileNotFoundError as e:
    print(f"ERROR CRÍTICO: {e}")
    # En una app real, podrías querer que esto detenga el arranque.

# --- Rutas de la Web ---
@app.route('/')
def index():
    """Sirve la página principal."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Endpoint para recibir datos del formulario y devolver la predicción.
    """
    try:
        data = request.get_json()
        # Mapeo de características en el ORDEN CORRECTO que tu modelo espera.
        # ¡IMPORTANTE! Este orden debe ser el mismo que usaste en el entrenamiento.
        # Basado en el dataset de fedesoriano, un orden común es:
        feature_order = [
            data['Age'], data['Sex'], data['ChestPainType'], data['RestingBP'],
            data['Cholesterol'], data['FastingBS'], data['RestingECG'], data['MaxHR'],
            data['ExerciseAngina'], data['Oldpeak'], data['ST_Slope']
        ]
        # Asegúrate de que los tipos de datos sean correctos (ej. 'Oldpeak' debe ser float)
        feature_order[0] = float(feature_order[0])
        feature_order[3] = float(feature_order[3])
        feature_order[4] = float(feature_order[4])
        feature_order[5] = float(feature_order[5])
        feature_order[7] = float(feature_order[7])
        feature_order[9] = float(feature_order[9])


        prediction, probability = predict(feature_order)

        if prediction is None:
            return jsonify({'success': False, 'error': probability}), 400

        # Guardar en historial
        save_prediction_to_db(data, prediction, probability)

        return jsonify({
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'message': 'Riesgo de enfermedad cardíaca' if prediction == 1 else 'Sin riesgo significativo'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history_route():
    """Endpoint para obtener el historial de predicciones."""
    try:
        history = get_prediction_history()
        return jsonify({'success': True, 'data': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Limpia todo el historial (¡Cuidado! Operación destructiva)."""
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Historial limpiado.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) # Cambia debug=False para producción