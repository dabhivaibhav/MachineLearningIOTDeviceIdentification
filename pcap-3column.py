from scapy.all import rdpcap, ARP, Ether, BOOTP, IP
import pandas as pd

# Function to extract MAC, IP, and Ethernet Type
def extract_mac_ip_ethertype(pcap_file):
    packets = rdpcap(pcap_file)
    mac_ip_mapping = []
    
    for packet in packets:
        if packet.haslayer(Ether):
            mac_address = packet[Ether].src
            eth_type = hex(packet[Ether].type)  # Ethernet type in hexadecimal
            ip_address = None
            
            if packet.haslayer(ARP) and packet[ARP].op == 2:  # ARP Reply
                ip_address = packet[ARP].psrc
            elif packet.haslayer(IP):  # Other IP-based packets
                ip_address = packet[IP].src
            elif packet.haslayer(BOOTP):  # DHCP Assignment
                ip_address = packet[BOOTP].yiaddr
                
            if mac_address and ip_address:
                mac_ip_mapping.append([mac_address, ip_address, eth_type])
    
    # Convert to DataFrame
    df = pd.DataFrame(mac_ip_mapping, columns=["MAC Address", "IP Address", "Ethernet Type"])
    return df

# Example usage
pcap_file = "./data/kasa-operation.pcap"  # Replace with your pcap file
mac_ip_df = extract_mac_ip_ethertype(pcap_file)

# Save the extracted data to a CSV file
mac_ip_df.to_csv("mac_ip_mapping.csv", index=False)

# Print results
print(mac_ip_df)
