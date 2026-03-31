# Compressed Federated Time-Series Forecasting

Open-source implementation of **compressed federated learning** for time-series forecasting using an LSTM model on the [Electricity Load Diagrams](https://huggingface.co/datasets/tulipa762/electricity_load_diagrams) dataset.

## Overview

This project demonstrates a federated learning setup where multiple clients collaboratively train a shared LSTM model **without sharing raw data**. To reduce communication overhead, model updates (deltas) are compressed with a custom **DeltaCompressor** before being sent to the server.

### Key Components

| Component | Description |
|---|---|
| **LSTM model** | Sequence model (`seq_len=10`) with a 64-unit LSTM layer followed by dense layers |
| **DeltaCompressor** | Sparsifies weight deltas via MAD + top-k thresholding, sign-encodes survivors, and gzip-compresses the payload |
| **Federated loop** | 100 rounds, 3 clients, 2 local epochs per round, mini-batch size 64 |
| **Evaluation** | Validation MAE logged every 10 rounds; final test MAE reported at end |

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python federated_lstm_forecasting.py
```

The script will:
1. Download the dataset automatically via Hugging Face `datasets`
2. Train the federated LSTM for 100 rounds
3. Print per-round upload size (KB) and validation MAE
4. Save aggregated uplink stats to `federated_lstm_compression_stats.csv`

## License

MIT
