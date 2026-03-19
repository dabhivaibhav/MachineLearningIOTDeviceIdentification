# IoT Network Device Identification via Machine Learning

Supervised ML pipeline that classifies IoT devices from raw network traffic (PCAP files) using Random Forest.  
Built as a graduate course project at Illinois State University.

## What it does

Network traffic from IoT devices is captured as PCAP files, converted to structured CSV features, and fed into a Random Forest classifier. The model identifies which type of IoT device generated a given traffic pattern.

SMOTE (Synthetic Minority Over-sampling Technique) was applied to handle class imbalance in the training data. Results are compared with and without SMOTE to demonstrate the impact on classification performance.

## Results

| Condition | Result |
|---|---|
| Without SMOTE | See confusion matrix below |
| With SMOTE | 100% classification accuracy across all device categories |

**Confusion Matrix — without SMOTE**  
![Confusion Matrix without SMOTE](16-09-23-Confusion%20Matrix%20without%20SMOTE.png)

**Confusion Matrix — with SMOTE**  
![Confusion Matrix with SMOTE](16-09-23-Confusion%20Matrix%20with%20SMOTE.png)

**Classification Report — with SMOTE**  
![Result with SMOTE](16-09-23-result%20with%20SMOTE.png)

## Pipeline

1. `pcap_to_csv.py` — extracts network features from raw PCAP captures into structured CSV
2. `listofdevice_txt_to_csv.py` — converts device label list to CSV format for labeling
3. `randomforestclassification.py` — trains and evaluates the Random Forest model
4. `randomforestclassification+reporting.py` — training with full classification report output

## Stack

Python · scikit-learn · pandas · NumPy · SMOTE (imbalanced-learn) · Wireshark / PCAP

## Dataset

Public network traffic dataset. Raw PCAP sample (`kasa-operation.pcap`) and processed CSV included in the repo.

## License

MIT — Copyright 2026 Vaibhav Dabhi
