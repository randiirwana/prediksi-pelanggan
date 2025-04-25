from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

app = Flask(__name__)

# Pastikan path file model dan scaler benar
model_path = os.path.join(os.path.dirname(__file__), "model_customer.h5") 
scaler_x_path = os.path.join(os.path.dirname(__file__), "scaler_X.pkl")
scaler_y_path = os.path.join(os.path.dirname(__file__), "scaler_Y.pkl")

try:
    # Load model dan scaler dengan error handling
    model = tf.keras.models.load_model(model_path)
    scaler_X = pickle.load(open(scaler_x_path, "rb"))
    scaler_Y = pickle.load(open(scaler_y_path, "rb"))
except Exception as e:
    print(f"Error loading model/scaler: {str(e)}")
    raise

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"]) 
def predict():
    try:
        # Ambil input dari form
        month_num = int(request.form["month"])
        year = month_num // 100
        month = month_num % 100
        
        # Prediksi untuk bulan yang diminta
        X_input = np.array([[month_num]])
        X_scaled = scaler_X.transform(X_input)
        pred_scaled = model.predict(X_scaled)
        prediction = scaler_Y.inverse_transform(pred_scaled)[0][0]
        
        # Buat data untuk grafik
        months = []
        predictions = []
        
        # Generate data points (2 bulan sebelum dan 2 bulan sesudah)
        for i in range(-2, 3):
            current_month = month_num + i
            current_year = current_month // 100
            current_month_in_year = current_month % 100
            
            if current_month_in_year > 12:
                current_year += 1
                current_month_in_year = 1
                current_month = current_year * 100 + current_month_in_year
            elif current_month_in_year < 1:
                current_year -= 1
                current_month_in_year = 12
                current_month = current_year * 100 + current_month_in_year
            
            # Format tahun dengan desimal untuk bulan
            decimal_year = float(f"{current_year}.{current_month_in_year:02d}")
            
            X = np.array([[current_month]])
            X_scaled = scaler_X.transform(X)
            pred = model.predict(X_scaled)
            pred_actual = scaler_Y.inverse_transform(pred)[0][0]
            
            months.append(decimal_year)
            predictions.append(float(pred_actual))
        
        # Format tahun dengan desimal untuk bulan yang dipilih
        selected_year = float(f"{year}.{month:02d}")
        
        return render_template(
            "index.html",
            prediction=float(prediction),
            month=selected_year,
            months=months,
            predictions=predictions
        )
        
    except ValueError as ve:
        return render_template("index.html", error="Format input tidak valid. Pastikan Anda memasukkan angka dengan format YYYYMM (contoh: 202401)")
    except Exception as e:
        return render_template("index.html", error=f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
