import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Step 1: Load and preprocess the data
# Assuming your CSV is named 'network_data.csv' and has no Device Category column
df = pd.read_csv('./data/kasa-operation.csv')

# Step 2: Aggregate data by Source MAC to represent unique devices
agg_df = df.groupby('Source MAC').agg({
    'Source IP': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0],  # Most frequent IP
    'Ethernet Type': lambda x: x.mode()[0],  # Most frequent Ethernet Type
    'Protocol': lambda x: x.value_counts().index[0],  # Most frequent protocol
    'Packet Length': 'mean'  # Average packet length
}).reset_index()

# Step 3: Feature preprocessing
# Encode Source MAC (categorical)
le_mac = LabelEncoder()
agg_df['Source MAC'] = le_mac.fit_transform(agg_df['Source MAC'])

# Convert Source IP to integer
def ip_to_int(ip):
    if ip == 'Unknown' or pd.isna(ip):
        return 0
    try:
        return int(''.join([f'{int(x):08b}' for x in ip.split('.')]), 2)
    except:
        return 0  # Handle malformed IPs
agg_df['Source IP'] = agg_df['Source IP'].apply(ip_to_int)

# One-hot encode Ethernet Type and Protocol
agg_df = pd.get_dummies(agg_df, columns=['Ethernet Type', 'Protocol'], prefix=['Eth', 'Proto'])

# Features for clustering
X = agg_df.drop(columns=['Source MAC'])  # Exclude Source MAC from clustering, use it for mapping later
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Clustering with K-Means (assume 2 clusters: IoT vs. Computer)
kmeans = KMeans(n_clusters=2, random_state=42)
agg_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Heuristic refinement of cluster labels
def assign_label(row):
    # Extract features from the aggregated row
    protocol = df[df['Source MAC'] == le_mac.inverse_transform([row['Source MAC']])[0]]['Protocol'].mode()[0]
    packet_length = row['Packet Length']
    oui = le_mac.inverse_transform([row['Source MAC']])[0][:8]  # First 3 bytes of MAC

    # Heuristic rules
    if protocol in ['TPLINK-SMARTHOME', 'DATA'] or oui == '36:cd:80':  # TP-Link OUI or IoT-specific protocol
        return 'Smart Home'
    elif packet_length > 1000 or protocol in ['TLS', 'DNS']:  # Large packets or computer-like protocols
        return 'Computer'
    elif packet_length < 100:  # Small packets typical of IoT control messages
        return 'Smart Home'
    else:
        return 'Computer' if row['Cluster'] == 1 else 'Smart Home'  # Default to cluster assignment

agg_df['Device Category'] = agg_df.apply(assign_label, axis=1)

# Step 6: Prepare data for supervised learning
# Encode Device Category for training
le_category = LabelEncoder()
agg_df['Device Category'] = le_category.fit_transform(agg_df['Device Category'])  # Smart Home=1, Computer=0

# Features for supervised model
X_supervised = agg_df.drop(columns=['Source MAC', 'Cluster', 'Device Category'])
y_supervised = agg_df['Device Category']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_supervised, y_supervised, test_size=0.2, random_state=42)

# Step 7: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Feature importance
importances = pd.DataFrame({
    'Feature': X_supervised.columns,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:")
print(importances.sort_values(by='Importance', ascending=False))

# Step 8: Predict on new data (example)
# Simulate new data (replace with actual new data if available)
new_data = X_test.iloc[:5]  # Take first 5 test samples as example
predictions = model.predict(new_data)
predicted_labels = le_category.inverse_transform(predictions)
print("\nExample Predictions for 5 devices:")
for i, (mac, label) in enumerate(zip(le_mac.inverse_transform(X_test.index[:5]), predicted_labels)):
    print(f"Device {i+1} (Source MAC: {mac}): {label}")

# Optional: Save the model and encoders for future use
import joblib
joblib.dump(model, 'device_classifier.pkl')
joblib.dump(le_mac, 'mac_encoder.pkl')
joblib.dump(le_category, 'category_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nModel and encoders saved.")