import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("16-09-23.csv")
df = df.sort_values("Timestamp")
df["Inter_Arrival_Time"] = df["Timestamp"].diff().fillna(0)

# Convert EtherType to numeric
df["EtherType"] = df["EtherType"].astype(str)
le_ethertype = LabelEncoder()
df["EtherType_enc"] = le_ethertype.fit_transform(df["EtherType"])

# New engineered features
df["Rolling_Avg_Size"] = df["Frame_Length"].rolling(window=5, min_periods=1).mean()
df["Rolling_Std_Size"] = df["Frame_Length"].rolling(window=5, min_periods=1).std().fillna(0)
df["Timestamp_rounded"] = df["Timestamp"].astype(int)
pps = df.groupby("Timestamp_rounded").size().reindex(df["Timestamp_rounded"], method='ffill').fillna(0)
df["Packets_per_Second"] = pps.values

# Define features and target (excluding MAC address)
X = df[["Frame_Length", "Inter_Arrival_Time", "EtherType_enc", "Rolling_Avg_Size", "Rolling_Std_Size", "Packets_per_Second"]]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(14, 10))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.title("Confusion Matrix - Device Classification (Without Src_MAC)", fontsize=12)
plt.tight_layout()
plt.show()
