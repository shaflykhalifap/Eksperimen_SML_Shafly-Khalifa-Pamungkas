import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    # Load dataset
    df = pd.read_csv(input_path)

    # Menangani Missing Values
    num_cols = ['Year', 'Engine Size', 'Mileage', 'Price']
    cat_cols = ['Brand', 'Model', 'Fuel Type', 'Transmission', 'Condition']

    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna("Unknown", inplace=True)

    # Deteksi duplikat
    df = df.drop_duplicates()

    # Mengubah Year menjadi Age
    df['Age'] = datetime.datetime.now().year - df['Year']

    #Deteksi dan menangani outlier menggunakan IQR
    numeric_features = ['Engine Size', 'Mileage', 'Age']
    for col in numeric_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Standarisasi fitur numerik
    scaler = StandardScaler()
    df[['Engine_scaled', 'Mileage_scaled', 'Age_scaled']] = scaler.fit_transform(
        df[['Engine Size', 'Mileage', 'Age']]
    )

    # Encoding fitur kategorikal
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])


    #Drop fitur yang sudah tak digunakan
    df.drop(['Year', 'Engine Size', 'Mileage', 'Age'], axis=1, inplace=True)

    # Save Preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed dataset saved to: {output_path}")


# Contoh penggunaan
if __name__ == "__main__":
    preprocess_data(
        input_path="namadataset_raw/car_price.csv",
        output_path="namadataset_preprocessing/car_price_preprocessed.csv"
    )
