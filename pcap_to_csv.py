import pyshark
import pandas as pd

# Function to extract relevant network details from a pcap file
def extract_pcap_features(pcap_file, output_csv):
    capture = pyshark.FileCapture(pcap_file)
    extracted_data = []
    
    for packet in capture:
        try:
            # Extract common network features
            protocol = packet.highest_layer if hasattr(packet, 'highest_layer') else 'Unknown'
            packet_length = packet.length if hasattr(packet, 'length') else 'Unknown'
            
            # Extract IP addresses
            src_ip = packet.ip.src if hasattr(packet, 'ip') else 'Unknown'
            dst_ip = packet.ip.dst if hasattr(packet, 'ip') else 'Unknown'
            
            # Extract MAC addresses (Ethernet layer)
            src_mac = packet.eth.src if hasattr(packet, 'eth') else 'Unknown'
            dst_mac = packet.eth.dst if hasattr(packet, 'eth') else 'Unknown'
            
            # Extract Ethernet type
            eth_type = packet.eth.type if hasattr(packet, 'eth') else 'Unknown'
            
            extracted_data.append([protocol, packet_length, src_ip, dst_ip, src_mac, dst_mac, eth_type])
        except AttributeError:
            continue
    
    capture.close()
    
    # Convert to DataFrame
    df = pd.DataFrame(extracted_data, columns=["Protocol", "Packet Length", "Source IP", "Destination IP", "Source MAC", "Destination MAC", "Ethernet Type"])
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Extraction complete. Data saved to {output_csv}")

# Example usage
pcap_file = "./data/kasa-operation.pcap"  # Replace with your actual pcap file
output_csv = "network_features.csv"
extract_pcap_features(pcap_file, output_csv)
