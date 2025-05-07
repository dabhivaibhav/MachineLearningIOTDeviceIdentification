import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("16-09-23.csv")

# Preprocessing

# Sort by timestamp
df = df.sort_values("Timestamp")

# Compute Inter-Arrival Time
df["Inter_Arrival_Time"] = df["Timestamp"].diff().fillna(0)

# Encode EtherType
le_ethertype = LabelEncoder()
df["EtherType"] = df["EtherType"].astype(str)
df["EtherType_enc"] = le_ethertype.fit_transform(df["EtherType"])

# Round timestamp to compute packet rate per second
df["Timestamp_rounded"] = df["Timestamp"].astype(int)
pps_df = (
    df.groupby(["Label", "Timestamp_rounded"]).size().reset_index(name="Packets_per_Second")
)
df = df.merge(pps_df, on=["Label", "Timestamp_rounded"], how="left")
df["Packets_per_Second"] = df["Packets_per_Second"].fillna(0)

# Rolling stats per device
window_size = 5
df["Rolling_Avg_Size"] = (
    df.groupby("Label")["Frame_Length"]
      .rolling(window=window_size, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)
df["Rolling_Std_Size"] = (
    df.groupby("Label")["Frame_Length"]
      .rolling(window=window_size, min_periods=1)
      .std()
      .fillna(0)
      .reset_index(level=0, drop=True)
)

# Features to use (excluding MACs)
features = ["Frame_Length", "Inter_Arrival_Time", "EtherType_enc", "Packets_per_Second", "Rolling_Avg_Size", "Rolling_Std_Size"]

# Apply per-device Isolation Forest
anomalies = pd.Series([False] * len(df), index=df.index)

for label in df["Label"].unique():
    device_df = df[df["Label"] == label].copy()
    if len(device_df) < 100:
        continue  # skip tiny devices

    contamination = max(0.0001, min(0.01, 5 / len(device_df)))
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(device_df[features])

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_scaled)
    preds = model.predict(X_scaled)  # -1 for anomaly

    anomalies[device_df.index] = preds == -1

# Add anomaly result to main df
df["Is_Spoofed"] = anomalies

# Display summary
spoofed = df[df["Is_Spoofed"] == True][["Timestamp", "Label", "Frame_Length", "EtherType", "Packets_per_Second"]]

print("\n🔐 Potential Spoofed Devices Detected:")
print(spoofed.head(10))

# Plot
plt.figure(figsize=(14, 6))
plt.scatter(df["Timestamp"], df["Frame_Length"], c="lightblue", s=8, label="Normal")
plt.scatter(df[df["Is_Spoofed"]]["Timestamp"], df[df["Is_Spoofed"]]["Frame_Length"], c="red", s=10, label="Spoofed")
plt.legend()
plt.title("Device Traffic Behavior (Spoofing Detection)")
plt.xlabel("Timestamp")
plt.ylabel("Frame Length")
plt.tight_layout()
plt.show()
