import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

df = pd.read_csv("Crop_recommendation.csv", on_bad_lines='skip')

df = df[df['N'] != 'N']  
df = df.dropna()   
 

df = df.sample(frac=0.2, random_state=42)

print("\n📄 Sample data from CSV:\n", df.head())

X = df.drop("label", axis=1)   
y = df["label"]                

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

# 🔻 Weak model (low depth, less trees)
model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print("\n🎯 Model Accuracy:", accuracy)

print("\n🧪 Enter Soil & Weather Info:")
n = int(input("Nitrogen (N%): "))
p = int(input("Phosphorus (P%): "))
k = int(input("Potassium (K%): "))
temp = float(input("Temperature (°C): "))
humidity = float(input("Humidity (%): "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall (mm%): "))

input_data = [[n, p, k, temp, humidity, ph, rainfall]]
predicted_crop = model.predict(input_data)[0]
print("\n✅ Recommended Crop:", predicted_crop)

conn = sqlite3.connect("crop_advisory.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS crop_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nitrogen INTEGER,
    phosphorus INTEGER,
    potassium INTEGER,
    temperature REAL,
    humidity REAL,
    ph REAL,
    rainfall REAL,
    predicted_crop TEXT
)
""")

cursor.execute("""
INSERT INTO crop_inputs (
    nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (n, p, k, temp, humidity, ph, rainfall, predicted_crop))

conn.commit()
conn.close()

print("🗃️ Data saved in crop_advisory.db")
