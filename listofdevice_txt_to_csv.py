import pandas as pd
import re

# Read the file content
with open("List_Of_Devices - Copy.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

data = []

# Loop through each line (skip empty and header lines)
for line in lines:
    if line.strip() == "" or "List of Devices" in line:
        continue

    # Split using 2+ spaces or tabs
    parts = re.split(r'\s{2,}|\t+', line.strip())
    
    if len(parts) >= 3:
        device_name = parts[0].strip()
        mac_address = parts[1].strip().lower()
        connection_type = parts[2].strip()
        data.append([device_name, mac_address, connection_type])
    else:
        print(f"⚠️ Skipped line due to parsing issue: {line.strip()}")

# Create DataFrame and save to CSV
df = pd.DataFrame(data, columns=["Device_Name", "MAC_Address", "Connection_Type"])
df.to_csv("List_Of_Devices.csv", index=False)
print("✅ Successfully converted to List_Of_Devices.csv")
