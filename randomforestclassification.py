import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# === Set up command-line arguments ===
parser = argparse.ArgumentParser(description="IoT Device Classification using Random Forest")
parser.add_argument("csv_file", help="Path to the input CSV file (extracted from PCAP)")
args = parser.parse_args()

# === Load CSV ===
df = pd.read_csv(args.csv_file)

# === Add Inter-Arrival Time ===
df = df.sort_values("Timestamp")
df["Inter_Arrival_Time"] = df["Timestamp"].diff().fillna(0)

# === Encode categorical features ===
df["EtherType"] = df["EtherType"].astype(str)
df["Src_MAC"] = df["Src_MAC"].astype(str)

le_ethertype = LabelEncoder()
df["EtherType_enc"] = le_ethertype.fit_transform(df["EtherType"])

le_mac = LabelEncoder()
df["Src_MAC_enc"] = le_mac.fit_transform(df["Src_MAC"])

# === Define features and target ===
X = df[["Frame_Length", "Inter_Arrival_Time", "EtherType_enc", "Src_MAC_enc"]]
y = df["Label"]

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(14, 10))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Confusion Matrix - Device Classification", fontsize=12)
plt.tight_layout()
plt.show()
