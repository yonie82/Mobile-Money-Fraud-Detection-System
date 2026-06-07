# Mobile Money Fraud Detection System Using Anomaly Detection
#Yonela Gingqi
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
#Transaction Dataset
# ==========================================

data = {
    "Amount": [50, 75, 100, 120, 80,150,220,300, 90, 60, 110, 95, 85,
               5000, 7000, 10000],
    "Hour": [9, 10, 11, 11, 12, 13,13, 14,14, 15, 16, 17, 18,
             2, 3, 1],
    "Transactions_Per_Day": [1, 2, 1, 2,1,2,1, 1, 2, 1, 3, 2, 1,
                             25, 30, 40]
}

df = pd.DataFrame(data)

print("Transaction Dataset:")
print(df)


# Preparing Data
# ==========================================

features = ["Amount", "Hour", "Transactions_Per_Day"]

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Training Anomaly Detection Model
# ==========================================

model = IsolationForest(
    contamination=0.1,   # Assume 10% fraud
    random_state=42
)

model.fit(X_scaled)


# Predict Fraud
# ==========================================

predictions = model.predict(X_scaled)

# Convert predictions
# 1 = Normal
# -1 = Fraud

df["Prediction"] = predictions

df["Prediction"] = df["Prediction"].replace({
    1: "Normal Transaction",
    -1: "Potential Fraud"
})

print("\nFraud Detection Results:")
print(df)

colors = df["Prediction"].map({
    "Normal Transaction": "blue",
    "Potential Fraud": "red"
})

plt.figure(figsize=(8,6))

plt.scatter(
    df["Amount"],
    df["Transactions_Per_Day"],
    c=colors,
    s=100
)
plt.xlabel("Transaction Amount")
plt.ylabel("Transactions Per Day")
plt.title("Mobile Money Fraud Detection")
plt.grid(True)
plt.show()

#Test New Transaction
# ==========================================

print("\nChecking New Transaction...")

new_transaction = pd.DataFrame({
    "Amount": [8500],
    "Hour": [2],
    "Transactions_Per_Day": [35]
})

new_scaled = scaler.transform(new_transaction)

result = model.predict(new_scaled)

if result[0] == -1:
    print("⚠ ALERT: Potential Fraud Detected!")
else:
    print("✓ Transaction Appears Normal")