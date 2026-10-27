import glob
import logging
import os
import pickle

import pandas as pd
from omegaconf import OmegaConf
from tokenization.vqvae_tokenizer import VQVAETokenizer


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs('data/tinystories', exist_ok=True)

config = OmegaConf.load('artifacts/tinystories/config.yaml')
tokenizer = VQVAETokenizer.from_config(config.tokenizer, base_dir='artifacts/tinystories')

parquet_files = glob.glob('data/tinystories_raw/data/train-*.parquet')[:1]
logger.info(f'Processing {len(parquet_files)} train file(s)')

context_length = 128
sequences = []
batch_size = 10000
total_saved = 0

for pf in parquet_files:
    df = pd.read_parquet(pf)
    total_rows = len(df)
    logger.info(f'Processing {pf}: {total_rows} rows')

    sequences = []
    row_count = 0

    for _i, row in enumerate(df.iterrows()):
        text = row['text']
        tokens = tokenizer.encode(text).tolist()

        if len(tokens) >= 128:
            for j in range(0, len(tokens) - 128 + 1, 64):
                sequences.append(tokens[j:j + 128])
        else:
            padded = tokens + [0] * (128 - len(tokens))
            sequences.append(padded[:128])

        row_count += 1

        # Save batch frequently
        if len(sequences) >= 10000:
            with open('data/tinystories/train.pkl', 'ab') as f:
                pickle.dump(sequences, f)
            total_saved += len(sequences)
            logger.info(f'Saved batch at row {row_count}, total sequences: {total_saved}')
            sequences = []

        # Progress
        if row_count % 10000 == 0:
            logger.info(f'  Row {row_count}/{total_rows}')

    # Save remaining
    if sequences:
        with open('data/tinystories/train.pkl', 'ab') as f:
                pickle.dump(sequences, f)
        total_saved += len(sequences)

logger.info(f'Done. Total sequences saved: {total_saved}')
import pickle
import os
from tokenization.vqvae_tokenizer import VQVAETokenizer
from omegaconf import OmegaConf
import pandas as pd

os.makedirs('data/tinystories', exist_ok=True)

config = OmegaConf.load('artifacts/tinystories/config.yaml')
tokenizer = VQVAETokenizer.from_config(config.tokenizer, base_dir='artifacts/tinystories')

parquet_files = glob.glob('data/tinystories_raw/data/train-*.parquet')[:1]
print(f'Processing {len(parquet_files)} train file(s)')

context_length = 128
sequences = []
batch_size = 10000
total_saved = 0

for pf in parquet_files:
    df = pd.read_parquet(pf)
    total_rows = len(df)
    print(f'Processing {pf}: {total_rows} rows')
    
    sequences = []
    row_count = 0
    
    for i, row in df.iterrows():
        text = row['text']
        tokens = tokenizer.encode(text).tolist()
        
        if len(tokens) >= 128:
            for j in range(0, len(tokens) - 128 + 1, 64):
                sequences.append(tokens[j:j + 128])
        else:
            padded = tokens + [0] * (128 - len(tokens))
            sequences.append(padded[:128])
        
        row_count += 1
        
        # Save batch frequently
        if len(sequences) >= 10000:
            with open('data/tinystories/train.pkl', 'ab') as f:
                pickle.dump(sequences, f)
            total_saved += len(sequences)
            print(f'Saved batch at row {row_count}, total sequences: {total_saved}')
            sequences = []
        
        # Progress
        if row_count % 10000 == 0:
            print(f'  Row {row_count}/{total_rows}')
    
    # Save remaining
    if sequences:
        with open('data/tinystories/train.pkl', 'ab') as f:
            pickle.dump(sequences, f)
        total_saved += len(sequences)

print(f'Done. Total sequences saved: {total_saved}')