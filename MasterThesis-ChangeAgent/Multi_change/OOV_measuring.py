import os, json, numpy as np, glob

list_path = r'.\data\LEVIR_MCI'
token_folder = r'.\data\LEVIR_MCI\tokens'
vocab_path = os.path.join(list_path, 'vocab.json')
val_list = os.path.join(list_path, 'val.txt')

with open(vocab_path,'r') as f:
    vocab = json.load(f)
END = vocab['<END>']

names = [l.strip() for l in open(val_list)]
tot_tokens = oov_tokens = 0
tot_caps = trunc_caps = 0
max_len = 46

for name in names:
    tok_file = os.path.join(token_folder, name.split('.')[0] + '.txt')
    if not os.path.exists(tok_file): continue
    caps = json.load(open(tok_file,'r'))  # list of token lists
    for tokens in caps:
        ids = [vocab.get(t, vocab['<UNK>']) for t in tokens]
        # account OOVs
        for t in tokens:
            tot_tokens += 1
            if t not in vocab:
                oov_tokens += 1
        # truncation check (dataset forces END)
        if len(ids) > max_len: trunc_caps += 1
        tot_caps += 1

print(f'OOV rate: {100.0*oov_tokens/max(1,tot_tokens):.2f}%')
print(f'Truncation rate (> {max_len}): {100.0*trunc_caps/max(1,tot_caps):.2f}%')