
# eval_baselines.py

import argparse
import json
import os
import time
import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

# Repo-local imports

from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def transplant_decoder_vocab_rows(decoder_module, checkpoint_decoder_sd, source_vocab_path, target_vocab_dict, log_fn=print):
    """
    Copy overlapping tokens' embedding and classifier rows from source checkpoint (original vocab)
    into the current decoder (augmented vocab) for the 'transplant' ablation.
    """
    if not source_vocab_path or not os.path.exists(source_vocab_path):
        log_fn("Vocab transplant: source_vocab_json not found; skipping.")
        return 0, 0

    with open(source_vocab_path, 'r') as f:
        src_tok2idx = json.load(f)  # token -> idx
    tgt_tok2idx = target_vocab_dict  # token -> idx

    common = set(src_tok2idx.keys()) & set(tgt_tok2idx.keys())
    if len(common) == 0:
        log_fn("Vocab transplant: no overlapping tokens; skipping.")
        return 0, 0

    old_emb = checkpoint_decoder_sd.get('vocab_embedding.weight', None)
    old_w = checkpoint_decoder_sd.get('wdc.weight', None)
    old_b = checkpoint_decoder_sd.get('wdc.bias', None)
    if old_emb is None or old_w is None or old_b is None:
        log_fn("Vocab transplant: missing embedding or classifier in checkpoint; skipping.")
        return 0, 0

    new_emb = decoder_module.vocab_embedding.weight.data
    new_w = decoder_module.wdc.weight.data
    new_b = decoder_module.wdc.bias.data

    copied = 0
    for tok in common:
        si = src_tok2idx[tok]
        ti = tgt_tok2idx[tok]
        if si < old_emb.shape[0] and ti < new_emb.shape[0]:
            new_emb[ti].copy_(old_emb[si]); copied += 1
        if si < old_w.shape[0] and ti < new_w.shape[0]:
            new_w[ti].copy_(old_w[si])
        if si < old_b.shape[0] and ti < new_b.shape[0]:
            new_b[ti].copy_(old_b[si])

    log_fn(f"Vocab transplant: copied {copied} overlapping tokens from source -> target.")
    return copied, len(common)


def load_checkpoint_with_logs(module, state_dict, name, strict=False, log_fn=print):
    """
    Load a state_dict into a module and print missing/unexpected keys for debugging.
    """
    try:
        missing, unexpected = module.load_state_dict(state_dict, strict=strict)
        # If strict=True, load_state_dict returns None. Handle both cases.
        if missing is None and unexpected is None:
            log_fn(f'{name}: loaded (strict={strict}).')
        else:
            log_fn(f'{name}: missing={len(missing)}, unexpected={len(unexpected)}')
            if len(missing) > 0:
                log_fn(f'{name} missing (sample): {missing[:10]}')
            if len(unexpected) > 0:
                log_fn(f'{name} unexpected (sample): {unexpected[:10]}')
    except Exception as e:
        log_fn(f'{name}: strict={strict} load failed: {e}. Retrying strict=False.')
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        log_fn(f'{name}: missing={len(missing)}, unexpected={len(unexpected)}')
        if len(missing) > 0:
            log_fn(f'{name} missing (sample): {missing[:10]}')
        if len(unexpected) > 0:
            log_fn(f'{name} unexpected (sample): {unexpected[:10]}')


def evaluate_caption_only(args):
    # Load vocab
    vocab_path = os.path.join(args.list_path, args.vocab_file + '.json')
    with open(vocab_path, 'r') as f:
        word_vocab = json.load(f)
    print(f'=>vocab: {args.vocab_file} (size={len(word_vocab)}, max_length={args.max_length})')

    # Data
    val_set = LEVIRCCDataset(
        args.data_folder, args.list_path, 'val',
        token_folder=args.token_folder, vocab_file=args.vocab_file,
        max_length=args.max_length, allow_unk=1
    )
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print(f'=>val size: {len(val_set)} items')

    # Models
    encoder = Encoder(args.network).cuda()
    encoder_trans = AttentiveEncoder(
        args.train_stage, args.n_layers,
        (args.feat_size, args.feat_size, args.encoder_dim),
        args.n_heads
    ).cuda()
    decoder = DecoderTransformer(
        encoder_dim=args.encoder_dim,
        feature_dim=args.feature_dim,
        vocab_size=len(word_vocab),
        max_lengths=args.max_length,
        word_vocab=word_vocab,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout
    ).cuda()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cuda')
    print(f'Load checkpoint: {args.checkpoint}')

    # Log encoder/encoder_trans loads
    enc_sd = ckpt.get('encoder_dict', {})
    enc_tr_sd = ckpt.get('encoder_trans_dict', {})
    dec_sd = ckpt.get('decoder_dict', {})

    load_checkpoint_with_logs(encoder, enc_sd, 'encoder', strict=False, log_fn=print)
    load_checkpoint_with_logs(encoder_trans, enc_tr_sd, 'encoder_trans', strict=False, log_fn=print)

    # Decoder: three profiles
    # - original_as_is: vocab matches; load whole decoder
    # - finetuned_aug: vocab matches; load whole decoder
    # - original_transplant: vocab differs; load shape-matching parts then transplant rows
    if args.profile in ['original_as_is', 'finetuned_aug']:
        missing, unexpected = decoder.load_state_dict(dec_sd, strict=False)
        print(f'decoder: missing={len(missing)}, unexpected={len(unexpected)}')
    else:
        # shape-matching copy
        model_sd = decoder.state_dict()
        skip = 0
        for k, v in dec_sd.items():
            if k in model_sd and model_sd[k].shape == v.shape and not k.startswith('vocab_embedding') and not k.startswith('wdc.'):
                model_sd[k].copy_(v)
            elif k in model_sd and model_sd[k].shape == v.shape:
                model_sd[k].copy_(v)
            else:
                skip += 1
        decoder.load_state_dict(model_sd, strict=False)
        print(f'decoder: skipped {skip} params due to vocab shape mismatch.')
        # transplant overlapping tokens from original vocab json
        if args.source_vocab_json and os.path.exists(args.source_vocab_json):
            transplant_decoder_vocab_rows(decoder, dec_sd, args.source_vocab_json, word_vocab, log_fn=print)
        else:
            print('Vocab transplant: source_vocab_json missing; skipped.')

    # Eval
    encoder.eval(); encoder_trans.eval(); decoder.eval()
    evaluator = Evaluator(num_class=3)
    references, hypotheses = [], []
    start = time.time()

    with torch.no_grad():
        for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _) in enumerate(
                tqdm(val_loader, desc=f'val_BEAM {args.beam_size}')):
            imgA = imgA.cuda(non_blocking=True)
            imgB = imgB.cuda(non_blocking=True)
            token_all = token_all.squeeze(0).cuda(non_blocking=True)  # batch=1

            feat1_list, feat2_list = encoder(imgA, imgB)
            feat1, feat2, seg_pre = encoder_trans(feat1_list, feat2_list)

            # Beam or greedy
            if args.beam_size > 1:
                seq = decoder.sample_beam(feat1, feat2, k=args.beam_size)
            else:
                seq = decoder.sample(feat1, feat2, k=1)

            # Build references/hypotheses (strip specials)
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in {
                word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
            references.append(img_tokens)
            pred_seq = [w for w in seq if w not in {
                word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
            hypotheses.append(pred_seq)

    elapsed = time.time() - start
    score = get_eval_score(references, hypotheses)
    print('Captioning_Validation:')
    print(f'Time: {elapsed:.3f}  BLEU-1: {score["Bleu_1"]:.5f}  BLEU-2: {score["Bleu_2"]:.5f}  '
          f'BLEU-3: {score["Bleu_3"]:.5f}  BLEU-4: {score["Bleu_4"]:.5f}  '
          f'Meteor: {score["METEOR"]:.5f}  Rouge: {score["ROUGE_L"]:.5f}  Cider: {score["CIDEr"]:.5f}')
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval baselines for LEVIR-MCI captioning')

    # Data
    parser.add_argument('--data_folder', default=r'C:\Users\nemade\Downloads\LEVIR-MCI-data\LEVIR-MCI-dataset\images')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/')
    parser.add_argument('--allow_unk', type=int, default=1)
    parser.add_argument('--beam_size', type=int, default=5)

    # Model/vocab profile
    # original_as_is: original MCI_model.pth with vocab_original + max_length 41 (no transplant)
    # finetuned_aug: your fine-tuned ckpt with augmented vocab + max_length 46
    # original_transplant: ablation (original ckpt transplanted to augmented vocab)
    parser.add_argument('--profile', choices=['original_as_is', 'finetuned_aug', 'original_transplant'], required=True)
    parser.add_argument('--vocab_file', default='vocab')
    parser.add_argument('--source_vocab_json', default='./data/LEVIR_MCI/vocab_original.json')
    parser.add_argument('--max_length', type=int, default=46)

    # Checkpoint
    parser.add_argument('--checkpoint', required=True)

    # Architecture
    parser.add_argument('--network', default='segformer-mit_b1')
    parser.add_argument('--train_stage', default='s2')
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--feat_size', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)

    args = parser.parse_args()

    # Apply profile defaults
    if args.profile == 'original_as_is':
        args.vocab_file = 'vocab_original'
        args.max_length = 41
    elif args.profile == 'finetuned_aug':
        args.vocab_file = 'vocab'
        args.max_length = 46
    elif args.profile == 'original_transplant':
        args.vocab_file = 'vocab'
        args.max_length = 46

    torch.cuda.set_device(0)
    evaluate_caption_only(args)