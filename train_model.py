import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

print("🧠 Initiating Model Training...")

# 1. Create Mock Patient Data (Age, Distance, Missed_Refills -> Churn)
data = {
    'Age': [45, 60, 25, 38, 70, 55, 30, 80],
    'Distance_km': [5, 20, 2, 15, 30, 8, 1, 25],
    'Missed_Refills': [0, 2, 0, 1, 3, 0, 0, 4],
    'Churn': [0, 1, 0, 0, 1, 0, 0, 1] 
}
df = pd.DataFrame(data)

# 2. Train the Model
X = df[['Age', 'Distance_km', 'Missed_Refills']]
y = df['Churn']
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 3. FREEZE THE MODEL
# This saves the trained algorithm as a physical file on your computer.
joblib.dump(model, 'patient_model.pkl')

print("✅ Model trained and saved as 'patient_model.pkl'.")