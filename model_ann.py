import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import pickle

try:
    # Load data
    data = pd.read_excel("Online Retail.xlsx")

    # Bersihkan data yang kosong
    data = data.dropna(subset=['CustomerID', 'InvoiceDate'])
    
    # Convert InvoiceDate ke datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    # Tambahkan kolom MonthNum dan Year
    data['MonthNum'] = data['InvoiceDate'].dt.month
    data['Year'] = data['InvoiceDate'].dt.year

    # Hitung jumlah customer unik per bulan (CustomerCount)
    monthly_customer = data.groupby(['Year', 'MonthNum'])['CustomerID'].nunique().reset_index()
    monthly_customer.columns = ['Year', 'MonthNum', 'CustomerCount']

    # Buat fitur gabungan Year dan Month
    monthly_customer['YearMonth'] = monthly_customer['Year']*12 + monthly_customer['MonthNum']

    # Ambil X dan Y dari hasil agregasi
    X = monthly_customer[['YearMonth']].values
    Y = monthly_customer['CustomerCount'].values

    # Normalisasi
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # Model ANN dengan arsitektur yang lebih kompleks
    model = Sequential([
        Dense(32, activation='relu', input_shape=(1,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(
        optimizer='adam', 
        loss='mean_squared_error',
        metrics=['mae']
    )

    # Training dengan early stopping
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, 
        Y_train,
        epochs=500,
        batch_size=32,
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluasi
    loss, mae = model.evaluate(X_test, Y_test)
    print(f"Loss: {loss:.4f}")
    print(f"MAE: {mae:.4f}")

    # Plot hasil training
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    # Membuat prediksi
    Y_pred = model.predict(X_scaled)

    # Plot hasil prediksi vs data aktual
    plt.figure(figsize=(10,6))
    plt.scatter(X_scaled, Y_scaled, color='blue', label='Data Aktual')
    plt.scatter(X_scaled, Y_pred, color='red', label='Prediksi ANN')
    plt.title('Hasil Prediksi ANN vs Data Aktual')
    plt.xlabel('Tahun')
    plt.ylabel('Jumlah Pelanggan')
    
    # Mengatur tampilan grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Mengatur legend
    plt.legend(bbox_to_anchor=(0.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('prediction_vs_actual.png')
    plt.close()

    # Simpan model
    model.save("model_customer.h5")

    # Simpan scaler_X
    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)

    # Simpan scaler_Y 
    with open("scaler_Y.pkl", "wb") as f:
        pickle.dump(scaler_Y, f)

except Exception as e:
    print(f"Terjadi error: {str(e)}")
