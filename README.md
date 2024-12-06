# ML-Algo

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("Gender Inequality Index.csv")
print(data.info())
print(data.describe())

# Visualisasi
numeric_data = data.select_dtypes(include=['number'])  # Selects columns with numeric types
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.show()

from sklearn.model_selection import train_test_split
target_column = 'HDI Rank (2021)'  
X = data.drop(target_column, axis=1)
y = data[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
data = pd.read_csv("Gender Inequality Index.csv")

# Dengan asumsi 'ISO3' adalah kolom dengan kode negara
X = data.drop(['HDI Rank (2021)', 'ISO3'], axis=1)  # Drop the target and the problematic column
y = data['HDI Rank (2021)']

# Membuat LabelEncoder object
label_encoder = LabelEncoder()

# Ulangi semua kolom di X untuk menemukan dan mengodekan tipe objek (string)
for col in X.select_dtypes(include=['object']).columns:
    X[col] = label_encoder.fit_transform(X[col])

# Hapus baris dengan nilai NaN dalam variabel target 'y'
X = X[y.notna()]  # Keep rows in X where 'y' is not NaN
y = y[y.notna()]  # Keep rows in 'y' where 'y' is not NaN


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

pip install streamlit

import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("Pentingnya Kesetaraan Gender")

# Input dari pengguna
umur = st.number_input("Umur", 0, 100)
akses_internet = st.selectbox("Akses Internet", ["Ya", "Tidak"])
Jenis_Gender = st.slider("Jenis Gender", 0, 10)

# Prediksi
if st.button("Prediksi"):
    data = pd.DataFrame([[umur, akses_internet, Jenis_Gender]], columns=["Umur", "Akses Internet", "Jenis Gender"])
    prediksi = model.predict(data)
    st.write(f"Hasil Prediksi: {prediksi}")

    !streamlit run app.py
