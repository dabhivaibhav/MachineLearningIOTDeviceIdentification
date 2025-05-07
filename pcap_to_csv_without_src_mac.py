import argparse
from scapy.all import rdpcap, Ether
import csv
import os
import pandas as pd

# Argument parsing
parser = argparse.ArgumentParser(description="Extract Layer 2 features from a PCAP file with device labels.")
parser.add_argument("pcap_file", help="Path to the input PCAP file")
parser.add_argument("--device_list_csv", default="List_Of_Devices.csv", help="Path to device list CSV file")
args = parser.parse_args()

# Output file name
base_filename = os.path.splitext(os.path.basename(args.pcap_file))[0]
args.output = f"{base_filename}.csv"

# Load device labels
device_df = pd.read_csv(args.device_list_csv)
device_df["MAC_Address"] = device_df["MAC_Address"].str.lower().str.strip()
mac_to_label = dict(zip(device_df["MAC_Address"], device_df["Device_Name"]))

# Read packets
packets = rdpcap(args.pcap_file)

# Extract features
with open(args.output, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Dst_MAC", "EtherType", "Frame_Length", "Label"])

    for pkt in packets:
        if Ether in pkt:
            eth = pkt[Ether]
            timestamp = pkt.time
            dst_mac = eth.dst.lower()
            ether_type = hex(eth.type)
            frame_length = len(pkt)

            label = mac_to_label.get(eth.src.lower(), "Unknown")
            writer.writerow([timestamp, dst_mac, ether_type, frame_length, label])

print(f"✅ Output saved to: {os.path.abspath(args.output)}")
