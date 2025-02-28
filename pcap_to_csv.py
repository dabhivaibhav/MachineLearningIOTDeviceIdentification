#!/usr/bin/env python3
import pyshark
import csv
import argparse

def pcap_to_csv(pcap_file, csv_file):
    # Open the capture (set keep_packets=False to avoid storing all packets in memory)
    capture = pyshark.FileCapture(pcap_file, keep_packets=False)
    
    # Define CSV columns – adjust or extend these as needed.
    fieldnames = [
        'frame_number', 'frame_time', 'frame_len',
        'eth_src', 'eth_dst', 'eth_type',
        'ip_version', 'ip_src', 'ip_dst', 'ip_hl', 'ip_len', 'ip_proto', 'ip_ttl',
        'tcp_srcport', 'tcp_dstport', 'tcp_seq', 'tcp_ack', 'tcp_len', 'tcp_flags'
    ]
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for packet in capture:
            row = {}
            
            # Frame layer extraction (Wireshark’s "Frame" details)
            try:
                row['frame_number'] = packet.frame_info.number
                row['frame_time']   = packet.frame_info.time
                row['frame_len']    = packet.frame_info.len
            except AttributeError:
                pass
            
            # Ethernet layer extraction
            if hasattr(packet, 'eth'):
                try:
                    row['eth_src']  = packet.eth.src
                    row['eth_dst']  = packet.eth.dst
                    row['eth_type'] = packet.eth.type
                except AttributeError:
                    pass
            
            # IP layer extraction
            if hasattr(packet, 'ip'):
                try:
                    row['ip_version'] = packet.ip.version
                    row['ip_src']     = packet.ip.src
                    row['ip_dst']     = packet.ip.dst
                    row['ip_hl']      = packet.ip.hdr_len
                    row['ip_len']     = packet.ip.len
                    row['ip_proto']   = packet.ip.proto
                    row['ip_ttl']     = packet.ip.ttl
                except AttributeError:
                    pass
            
            # TCP layer extraction
            if hasattr(packet, 'tcp'):
                try:
                    row['tcp_srcport'] = packet.tcp.srcport
                    row['tcp_dstport'] = packet.tcp.dstport
                    row['tcp_seq']     = packet.tcp.seq
                    row['tcp_ack']     = packet.tcp.ack
                    row['tcp_len']     = packet.tcp.len
                    row['tcp_flags']   = packet.tcp.flags
                except AttributeError:
                    pass
            
            writer.writerow(row)
    
    capture.close()
    print(f"CSV file '{csv_file}' has been created from '{pcap_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CSV from a pcap file with detailed layer information"
    )
    parser.add_argument("pcap_file", help="Path to the input pcap file")
    parser.add_argument("csv_file", help="Path for the output CSV file")
    args = parser.parse_args()
    
    pcap_to_csv(args.pcap_file, args.csv_file)
