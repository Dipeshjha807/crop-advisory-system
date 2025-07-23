
import pandas as pd
import sqlite3
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and clean the dataset
df = pd.read_csv("Crop_recommendation.csv", on_bad_lines='skip')
df = df[df['N'] != 'N']
df = df.dropna()

# Splitting the data
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize the GUI
root = Tk()
root.title("🌾 Crop Recommendation System")
root.geometry("400x500")
root.configure(bg="#e6ffe6")

Label(root, text="Enter Soil & Weather Info", font=("Arial", 14, "bold"), bg="#e6ffe6").pack(pady=10)

fields = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", 
          "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (mm)"]

entries = []

for field in fields:
    Label(root, text=field, bg="#e6ffe6").pack()
    entry = Entry(root, font=("Arial", 12))
    entry.pack(pady=2)
    entries.append(entry)

def predict_crop():
    try:
        inputs = [float(entry.get()) for entry in entries]
        predicted_crop = model.predict([inputs])[0]

        # Store in SQLite
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
        cursor.execute("INSERT INTO crop_inputs (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, predicted_crop) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (*inputs, predicted_crop))
        conn.commit()
        conn.close()

        messagebox.showinfo("✅ Crop Recommended", f"🌿 You should grow: {predicted_crop}")
    except ValueError:
        messagebox.showerror("❌ Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("❌ Error", str(e))

Button(root, text="Predict Crop", font=("Arial", 12), bg="#4CAF50", fg="white", command=predict_crop).pack(pady=20)

root.mainloop()
