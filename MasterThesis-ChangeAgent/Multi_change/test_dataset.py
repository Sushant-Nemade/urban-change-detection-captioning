#!/usr/bin/env python3
"""
test_dataset.py: Standalone script to test LEVIRCCDataset loading and batching.
Run this before full training to verify no errors in data loading, tokenization, or truncation.

Usage: python test_dataset.py
Adjust paths at the top if needed.
"""

import os
import sys
import warnings
import json
import numpy as np
import torch
import traceback
from torch.utils.data import DataLoader

# Suppress PyTorch UserWarning about attention masks (harmless deprecation)
warnings.filterwarnings("ignore", category=UserWarning, message="Support for mismatched key_padding_mask and attn_mask")

# FIXED: Robust path setup for CLI and IDE
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)  # Add root (Multi_change/) to path

# Add data/ to path for direct access to LEVIR_MCI.py (CLI-compatible)
data_dir = os.path.join(current_dir, 'data')
if os.path.exists(data_dir):
    sys.path.insert(0, data_dir)  # Now 'from LEVIR_MCI import ...' works

print(f"Debug: Working dir: {current_dir}")
print(f"Debug: sys.path[0]: {sys.path[0]}")
print(f"Debug: data_dir added: {data_dir}")

# FIXED: Direct imports (NO 'Multi_change.' prefix) - works from root
try:
    from LEVIR_MCI import LEVIRCCDataset  # Direct from added data/ path
    from preprocess_data import encode    # Direct from root
    print("✓ Successfully imported LEVIRCCDataset and encode.")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Debug: Files checked - LEVIR_MCI.py in data/? preprocess_data.py in root?")
    # Fallback: Try subdir import if data/ not added
    try:
        sys.path.insert(0, current_dir)  # Ensure root first
        from data.LEVIR_MCI import LEVIRCCDataset
        from preprocess_data import encode
        print("✓ Fallback import successful.")
    except ImportError as e2:
        print(f"✗ Fallback failed: {e2}")
        print("Fix: Add empty 'data/__init__.py' or check file locations.")
        sys.exit(1)

def main():
    # ================= CONFIGURE PATHS HERE =================
    # Your existing paths (keep as-is - they work)
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(project_root)
    data_folder = r'C:\Users\nemade\Downloads\LEVIR-MCI-data\LEVIR-MCI-dataset\images'  # UPDATE THIS PATH
    list_path = os.path.join(project_root, 'data', 'LEVIR_MCI')  # Contains train.txt, val.txt, vocab.json
    token_folder = os.path.join(project_root, 'data', 'LEVIR_MCI', 'tokens')
    vocab_file = 'vocab'
    split = 'train'
    max_length = 41
    allow_unk = 1
    batch_size = 2
    num_workers = 0  # Use 0 for Windows compatibility
    # =========================================================

    print(f"Testing dataset loading for split: {split}")
    print(f"Project root: {project_root}")
    print(f"Data folder: {data_folder}")
    print(f"List path: {list_path}")
    print(f"Token folder: {token_folder}")
    print(f"Max length: {max_length}, Batch size: {batch_size}")

    # Verify critical paths exist
    list_file = os.path.join(list_path, f'{split}.txt')
    vocab_path = os.path.join(list_path, f'{vocab_file}.json')

    if not os.path.exists(list_file):
        print(f"✗ List file not found: {list_file}")
        print("Make sure train.txt exists in data/LEVIR_MCI/")
        return

    if not os.path.exists(vocab_path):
        print(f"✗ Vocab file not found: {vocab_path}")
        print("Run preprocess_data.py to generate vocab.json")
        return

    # Step 1: Load vocabulary
    try:
        with open(vocab_path, 'r') as f:
            word_vocab = json.load(f)
        # Create reverse vocab for decoding
        reverse_vocab = {v: k for k, v in word_vocab.items()}
        print(f"✓ Loaded vocab with {len(word_vocab)} tokens")
    except Exception as e:
        print(f"✗ Vocab loading error: {e}")
        traceback.print_exc()
        return

    # Step 2: Initialize dataset
    try:
        dataset = LEVIRCCDataset(
            data_folder=data_folder,
            list_path=list_path,
            split=split,
            token_folder=token_folder,
            vocab_file=vocab_file,
            max_length=max_length,
            allow_unk=allow_unk
        )
        print(f"✓ Dataset initialized successfully. Total samples: {len(dataset)}")
    except Exception as e:
        print(f"✗ Dataset init error: {e}")
        print("\nCommon issues:")
        print("  - Make sure your data_folder path points to the LEVIR-MCI images directory")
        print("  - Verify the directory structure: data_folder/train/A/, data_folder/train/B/, data_folder/train/label/")
        print("  - Check that train.txt contains valid image filenames")
        traceback.print_exc()
        return

    # Step 3: Create DataLoader
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Keep deterministic for testing
            num_workers=num_workers,
            pin_memory=False  # Disable for testing
        )
        print(f"✓ DataLoader created with batch_size={batch_size}, num_workers={num_workers}")
    except Exception as e:
        print(f"✗ DataLoader error: {e}")
        traceback.print_exc()
        return

    # Step 4: Load one batch and inspect
    try:
        print("\nLoading first batch...")
        batch_iter = iter(dataloader)
        batch = next(batch_iter)
        print("✓ Batch loaded successfully without errors!")

        # Unpack (imgA, imgB, seg_label, token_all, token_all_len, token, token_len, names)
        imgA, imgB, seg_label, token_all, token_all_len, token, token_len, names = batch

        # Print shapes
        print("\nBatch Details:")
        print(f"  - imgA shape: {imgA.shape}")
        print(f"  - imgB shape: {imgB.shape}")
        print(f"  - seg_label shape: {seg_label.shape}")
        print(f"  - token_all shape: {token_all.shape}")
        print(f"  - token_all_len shape: {token_all_len.shape}")
        print(f"  - token shape: {token.shape}")
        print(f"  - token_len shape: {token_len.shape}")
        print(f"  - names: {names}")

        # Token validation
        token_len_np = token_len.numpy() if hasattr(token_len, 'numpy') else token_len
        if torch.is_tensor(token_len_np):
            token_len_np = token_len_np.numpy()
        max_token_len = np.max(token_len_np)
        print(f"\nToken Length Validation:")
        print(f"  - Max selected token length: {max_token_len} (should be ≤ {max_length})")
        print(f"  - All selected lengths: {token_len_np.flatten()}")
        if max_token_len > max_length:
            print("  ⚠ WARNING: Tokens exceed max_length!")
        else:
            print("  ✓ All within max_length - good!")

        # First sample details
        first_idx = 0
        print(f"\nFirst Sample:")
        print(f"  - Name: {names[first_idx]}")
        if len(token_all.shape) > 1:
            print(f"  - Num captions: {token_all.shape[1]}")
        print(f"  - Selected caption length: {token_len_np[first_idx] if len(token_len_np) > first_idx else 'N/A'}")
        print(f"  - Sample tokens (first 10): {token[first_idx][:10]}")

        # Decode sample (optional, using reverse_vocab)
        print("  - Decoded first 5 non-padding words:")
        sample_tokens = token[first_idx].numpy() if hasattr(token, 'numpy') else token[first_idx]
        decoded = []
        for tok_id in sample_tokens[:10]:
            if tok_id != 0:  # Skip pad
                word = reverse_vocab.get(int(tok_id), '<unk>')
                if word not in ['<pad>', '<sos>', '<eos>', '<unk>']:
                    decoded.append(word)
                if len(decoded) >= 5:
                    break
        print(f"    {' '.join(decoded) or 'All special/padding'}")

        print("\n🎉 Test PASSED! Ready for training.")

    except StopIteration:
        print("⚠ No data available in the dataset.")
        return
    except Exception as e:
        print(f"✗ Batch error: {e}")
        traceback.print_exc()
        print("\nDebugging tips:")
        print("  - Verify your LEVIR-MCI dataset images are in the correct directory structure")
        print("  - Check that image files exist in data_folder/train/A/, data_folder/train/B/, data_folder/train/label/")
        print("  - Ensure token files exist in the tokens folder")
        return

    # Step 5: Single sample test
    print("\nTesting single sample...")
    try:
        single_sample = dataset[0]
        print(f"✓ Loaded single sample: {len(single_sample)} elements")
        print(f"  - imgA shape: {single_sample[0].shape}")
        print(f"  - token shape: {single_sample[5].shape}")
        print(f"  - token_len: {single_sample[6]}")
    except Exception as e:
        print(f"✗ Single sample error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
