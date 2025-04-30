import argparse
from scapy.all import rdpcap, Ether
import csv
import os
import re

parser = argparse.ArgumentParser(description="Extract Layer 2 features from a PCAP file with device labels.")
parser.add_argument("pcap_file", help="Path to the input PCAP file")
parser.add_argument("--device_list", default="List_Of_Devices - Copy.txt", help="Path to device list file")
args = parser.parse_args()

base_filename = os.path.splitext(os.path.basename(args.pcap_file))[0]
args.output = f"{base_filename}.csv"

mac_to_label = {}
with open(args.device_list, 'r') as f:
    lines = f.readlines()

for line in lines[1:]: 
    parts = re.split(r'\s{2,}', line.strip())
    if len(parts) >= 2:
        label = parts[0].strip()
        mac = parts[1].strip().lower()
        mac_to_label[mac] = label

packets = rdpcap(args.pcap_file)

with open(args.output, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Src_MAC", "Dst_MAC", "EtherType", "Frame_Length", "Label"])

    for pkt in packets:
        if Ether in pkt:
            eth = pkt[Ether]
            timestamp = pkt.time
            src_mac = eth.src.lower()
            dst_mac = eth.dst.lower()
            ether_type = hex(eth.type)
            frame_length = len(pkt)

            label = mac_to_label.get(src_mac) or mac_to_label.get(dst_mac) or "Unknown"
            writer.writerow([timestamp, src_mac, dst_mac, ether_type, frame_length, label])

print(f"✅ Output saved to: {os.path.abspath(args.output)}")