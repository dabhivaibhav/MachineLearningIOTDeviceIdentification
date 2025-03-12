import pyshark
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Function to extract relevant network details from a pcap file
def extract_pcap_features(pcap_file):
    output_csv = pcap_file.replace(".pcap", ".csv")
    capture = pyshark.FileCapture(pcap_file)
    extracted_data = []
    
    for packet in capture:
        try:
            # Extract common network features
            protocol = packet.highest_layer if hasattr(packet, 'highest_layer') else 'Unknown'
            packet_length = int(packet.length) if hasattr(packet, 'length') else 0
            
            # Extract MAC addresses and IP addresses
            src_mac = packet.eth.src if hasattr(packet, 'eth') else 'Unknown'
            dst_mac = packet.eth.dst if hasattr(packet, 'eth') else 'Unknown'
            src_ip = packet.ip.src if hasattr(packet, 'ip') else 'Unknown'
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else 'Unknown'
            
            # Extract Ethernet type
            eth_type = packet.eth.type if hasattr(packet, 'eth') else 'Unknown'
            
            extracted_data.append([protocol, packet_length, src_mac, dst_mac, src_ip, dst_ip, eth_type])
        except AttributeError:
            continue
    
    capture.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(extracted_data, columns=["Protocol", "Packet Length", "Source MAC", "Destination MAC", "Source IP", "Destination IP", "Ethernet Type"])
    
    # Define categories for classification: Smart Home, Computer, IoT
    device_categories = {
        "Smart Home": ["36:cd:80:d9:58:3d", "6c:5a:b0:e6:03:cd"],
        "Computer": ["b8:27:eb:59:60:d1"],
        "IoT": []  # Add more MAC addresses as necessary
    }
    
    # Map MAC Address to categories
    def categorize_device(mac):
        for category, devices in device_categories.items():
            if mac in devices:
                return category
        return "Unknown"
    
    df["Device Category"] = df["Source MAC"].apply(categorize_device)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Data saved to {output_csv}")
    
    return df, output_csv

# Function to train a RandomForestClassifier and visualize classification
def train_and_plot_classification(data_file):
    df = pd.read_csv(data_file)
    
    # Encode categorical features
    le_protocol = LabelEncoder()
    le_eth = LabelEncoder()
    le_category = LabelEncoder()
    
    df["Protocol"] = le_protocol.fit_transform(df["Protocol"])
    df["Ethernet Type"] = le_eth.fit_transform(df["Ethernet Type"])
    df["Device Category"] = le_category.fit_transform(df["Device Category"])
    
    # Define features and labels
    X = df[["Protocol", "Packet Length", "Ethernet Type"]]
    y = df["Device Category"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Generate plot
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df["Packet Length"], df["Protocol"], c=df["Device Category"], cmap='viridis', alpha=0.7, edgecolors='k')
    
    # Add dynamic legend based on unique labels
    unique_labels = np.unique(y)
    legend_labels = le_category.inverse_transform(unique_labels).tolist()  # Convert to list to avoid ambiguity
    plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.xlabel("Packet Length")
    plt.ylabel("Protocol")
    plt.title("Classification of Device Categories")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from a pcap file and classify device types.")
    parser.add_argument("pcap_file", help="Path to the pcap file")
    args = parser.parse_args()
    
    df_extracted, output_csv = extract_pcap_features(args.pcap_file)
    train_and_plot_classification(output_csv)
