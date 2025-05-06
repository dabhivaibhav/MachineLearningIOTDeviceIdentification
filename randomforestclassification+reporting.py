import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# === Load and prepare dataset ===
df = pd.read_csv("16-09-23.csv")
df = df.sort_values("Timestamp")
df["Inter_Arrival_Time"] = df["Timestamp"].diff().fillna(0)
df["EtherType"] = df["EtherType"].astype(str)
df["Src_MAC"] = df["Src_MAC"].astype(str)

# Encode features
le_ethertype = LabelEncoder()
df["EtherType_enc"] = le_ethertype.fit_transform(df["EtherType"])
le_mac = LabelEncoder()
df["Src_MAC_enc"] = le_mac.fit_transform(df["Src_MAC"])

# === Preserve df_test for spoofing logic ===
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Extract features/labels
X = df[["Frame_Length", "Inter_Arrival_Time", "EtherType_enc", "Src_MAC_enc"]]
y = df["Label"]

# Apply SMOTE to resample features
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split from resampled data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# === Train classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict and evaluate ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(14, 10))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Confusion Matrix - Device Classification", fontsize=12)
plt.tight_layout()
plt.show()

# === Trusted Device MACs ===
trusted_device_mac_map = {
    "Smart Things": "d0:52:a8:00:67:5e",
    "Amazon Echo": "44:65:0d:56:cc:d3",
    "Netatmo Welcome": "70:ee:50:18:34:43",
    "TP-Link Day Night Cloud camera": "f4:f2:6d:93:51:f1",
    "Samsung SmartCam": "00:16:6c:ab:6b:88",
    "Dropcam": "30:8c:fb:2f:e4:b2",
    "Insteon Camera": "00:62:6e:51:27:2e",
    "Unknown": "e8:ab:fa:19:de:4f",
    "Withings Smart Baby Monitor": "00:24:e4:11:18:a8",
    "Belkin Wemo switch": "ec:1a:59:79:f4:89",
    "TP-Link Smart plug": "50:c7:bf:00:56:39",
    "iHome": "74:c6:3b:29:d7:1d",
    "Belkin wemo motion sensor": "ec:1a:59:83:28:11",
    "NEST Protect smoke alarm": "18:b4:30:25:be:e4",
    "Netatmo weather station": "70:ee:50:03:b8:ac",
    "Withings Smart scale": "00:24:e4:1b:6f:96",
    "Blipcare Blood Pressure meter": "74:6a:89:00:2e:25",
    "Withings Aura smart sleep sensor": "00:24:e4:20:28:c6",
    "Light Bulbs LiFX Smart Bulb": "d0:73:d5:01:83:08",
    "Triby Speaker": "18:b7:9e:02:20:44",
    "PIX-STAR Photo-frame": "e0:76:d0:33:bb:85",
    "HP Printer": "70:5a:0f:e4:9b:c0",
    "Samsung Galaxy Tab": "08:21:ef:3b:fc:e3",
    "Nest Dropcam": "30:8c:fb:b6:ea:45",
    "Android Phone 1": "40:f3:08:ff:1e:da",
    "Laptop": "74:2f:68:81:69:42",
    "MacBook": "ac:bc:32:d4:6f:2f",
    "Android Phone 2": "b4:ce:f6:a7:a3:c2",
    "IPhone": "d0:a6:37:df:a1:e1",
    "MacBook/Iphone": "f4:5c:89:93:cc:85",
    "TPLink Router Bridge LAN (Gateway)": "14:cc:20:51:33:ea"
}


# === Spoofing Detection Logic ===
df_test = df_test.copy()
df_test["Predicted_Label"] = model.predict(df_test[["Frame_Length", "Inter_Arrival_Time", "EtherType_enc", "Src_MAC_enc"]])
df_test["Src_MAC"] = df_test["Src_MAC"].str.lower()
df_test["OUI"] = df_test["Src_MAC"].str[:8]
df_test["Is_Known_MAC"] = df_test["Src_MAC"].isin(trusted_device_mac_map.values())

df_test["Expected_MAC"] = df_test["Predicted_Label"].map(trusted_device_mac_map)
df_test["Expected_OUI"] = df_test["Predicted_Label"].map(lambda x: trusted_device_mac_map.get(x, "")[:8])
df_test["MAC_Match"] = df_test["Src_MAC"] == df_test["Expected_MAC"]
df_test["OUI_Match"] = df_test["OUI"] == df_test["Expected_OUI"]
df_test["Is_Spoofed"] = (~df_test["MAC_Match"]) & (df_test["OUI_Match"])

# === Spoofing Report ===
spoofed_summary = df_test[df_test["Is_Spoofed"]].groupby("Predicted_Label").size().sort_values(ascending=False)
print("\n🔒 Detected Spoofing by OUI Match + MAC Mismatch:")
print(spoofed_summary)

unknown_macs = df_test[df_test["Is_Known_MAC"] == False]["Predicted_Label"].value_counts()
print("\n🚨 Devices using Unknown MAC addresses:")
print(unknown_macs)

misclassified = df_test[df_test["Predicted_Label"] != df_test["Label"]]
print("\n⚠️ Misclassified Devices (could indicate compromise or learning gap):")
print(misclassified[["Src_MAC", "Label", "Predicted_Label"]].head(10))
