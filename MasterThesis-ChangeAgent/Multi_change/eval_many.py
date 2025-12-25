# eval_many.py
import os
import time
import json
import argparse
import csv
import numpy as np
import torch
from tqdm import tqdm
from torch.utils import data

from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import get_eval_score
from utils_tool.metrics import Evaluator

def load_ckpt_list(path):
    paths = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if s.lower().endswith('.pth'):
                paths.append(s)
    return paths

def load_vocab(list_path, vocab_name):
    with open(os.path.join(list_path, vocab_name + '.json'), 'r') as f:
        return json.load(f)

def evaluate_one(args, ckpt_path, beam_size=5, det_tta=0, split='test',
                 aug_vocab=None, orig_vocab=None, aug_size=None, orig_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read checkpoint on CPU and infer vocab size + max_length from decoder buffers
    checkpoint = torch.load(ckpt_path, map_location=device)
    dec_sd = checkpoint['decoder_dict']
    ck_vocab_size = dec_sd['wdc.weight'].shape[0]
    ck_max_len = dec_sd['position_encoding.pe'].shape[0]

    # Decide which vocab to use (augmented vs original)
    # If exact match to augmented/original size, pick that; else fallback to augmented.
    if aug_size is not None and ck_vocab_size == aug_size:
        word_vocab = aug_vocab
        vocab_name_used = args.vocab_file
    elif orig_size is not None and ck_vocab_size == orig_size:
        word_vocab = orig_vocab
        vocab_name_used = args.orig_vocab_file
    else:
        # Fallback: use augmented vocab (safe; still loads decoder weights strictly)
        word_vocab = aug_vocab
        vocab_name_used = args.vocab_file
        print(f'Warning: ckpt vocab_size={ck_vocab_size} does not match aug({aug_size}) or orig({orig_size}); using {vocab_name_used}')

    specials = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}

    # Build models matching the checkpoint’s decoder settings
    encoder = Encoder(args.network).eval().to(device)
    encoder_trans = AttentiveEncoder(None, args.n_layers, [args.feat_size, args.feat_size, args.encoder_dim],
                                     args.n_heads, args.dropout).eval().to(device)
    decoder = DecoderTransformer(args.encoder_dim, args.feature_dim, len(word_vocab), ck_max_len,
                                 word_vocab, args.n_heads, args.decoder_n_layers, args.dropout).eval().to(device)

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
    decoder.load_state_dict(dec_sd, strict=True)

    # Build dataset with the same vocab and max_length used by this checkpoint
    ds = LEVIRCCDataset(args.data_folder, args.list_path, split,
                        args.token_folder, vocab_name_used, ck_max_len, args.allow_unk)
    loader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    evaluator = Evaluator(num_class=3)
    references, hypotheses = [], []

    def forward_once(imgA, imgB):
        feat1_list, feat2_list = encoder(imgA, imgB)
        feat1, feat2, seg_pre = encoder_trans(feat1_list, feat2_list)
        return feat1, feat2, seg_pre

    t0 = time.time()
    with torch.no_grad():
        for _, (imgA, imgB, seg_label, token_all, _, _, _, _) in enumerate(
                tqdm(loader, desc=f'eval {os.path.basename(ckpt_path)} ({split})')):
            imgA = imgA.to(device, non_blocking=True)
            imgB = imgB.to(device, non_blocking=True)
            token_all = token_all.squeeze(0).to(device, non_blocking=True)

            f1, f2, seg_pre = forward_once(imgA, imgB)

            if det_tta == 1:
                segs = [seg_pre]
                _, _, seg_h = forward_once(torch.flip(imgA, dims=[-2]), torch.flip(imgB, dims=[-2]))
                segs.append(torch.flip(seg_h, dims=[-2]))
                _, _, seg_v = forward_once(torch.flip(imgA, dims=[-1]), torch.flip(imgB, dims=[-1]))
                segs.append(torch.flip(seg_v, dims=[-1]))
                seg_pre = torch.mean(torch.stack(segs, dim=0), dim=0)

            if beam_size > 1:
                seq = decoder.sample_beam(f1, f2, k=beam_size)
            else:
                seq = decoder.sample(f1, f2, k=1)

            # detection
            pred_seg = seg_pre.detach().cpu().numpy()
            seg_label_np = seg_label.numpy()
            pred_seg = np.argmax(pred_seg, axis=1)
            evaluator.add_batch(seg_label_np, pred_seg)

            # captions
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in specials], img_token))
            references.append(img_tokens)
            if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
                seq = seq[0]
            pred_seq = [w for w in seq if w not in specials]
            hypotheses.append(pred_seq)

    elapsed = time.time() - t0

    # detection metrics
    Acc_seg = evaluator.Pixel_Accuracy()
    Acc_class_seg = evaluator.Pixel_Accuracy_Class()
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()

    # caption metrics
    score = get_eval_score(references, hypotheses)
    return {
        'checkpoint': ckpt_path,
        'split': split,
        'vocab_used': vocab_name_used,
        'ck_vocab_size': ck_vocab_size,
        'ck_max_len': ck_max_len,
        'beam_size': beam_size,
        'det_tta': det_tta,
        'time_sec': round(elapsed, 2),
        'Acc_seg': Acc_seg,
        'Acc_class_seg': Acc_class_seg,
        'mIoU': mIoU_seg,
        'FWIoU': FWIoU_seg,
        'BLEU_1': score['Bleu_1'],
        'BLEU_2': score['Bleu_2'],
        'BLEU_3': score['Bleu_3'],
        'BLEU_4': score['Bleu_4'],
        'METEOR': score['METEOR'],
        'ROUGE_L': score['ROUGE_L'],
        'CIDEr': score['CIDEr']
    }

def main():
    parser = argparse.ArgumentParser()

    # data/config (match training)
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--list_path', required=True)
    parser.add_argument('--token_folder', required=True)
    parser.add_argument('--vocab_file', default='vocab', help='augmented vocab file name (without .json)')
    parser.add_argument('--orig_vocab_file', default='vocab_original', help='original vocab file name (without .json)')
    parser.add_argument('--max_length', type=int, default=46)  # default for augmented; per-ckpt override is automatic
    parser.add_argument('--allow_unk', type=int, default=1)
    parser.add_argument('--network', default='segformer-mit_b1')
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--feat_size', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--workers', type=int, default=0)

    # inputs
    parser.add_argument('--ckpt_list', required=True, help='text file: one .pth path per line')
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--det_tta', type=int, default=0)
    parser.add_argument('--split', default='test', choices=['test', 'val'], help='dataset split to evaluate')
    parser.add_argument('--out_csv', default='results_eval_many.csv')

    args = parser.parse_args()

    # Load both vocabs once to know sizes
    aug_vocab = load_vocab(args.list_path, args.vocab_file)
    orig_vocab = load_vocab(args.list_path, args.orig_vocab_file)
    aug_size = len(aug_vocab)
    orig_size = len(orig_vocab)

    ckpts = load_ckpt_list(args.ckpt_list)
    if not ckpts:
        print(f'No checkpoints found in {args.ckpt_list}.')
        return

    print(f'Found {len(ckpts)} checkpoints.')
    rows = []
    for ck in ckpts:
        res = evaluate_one(args, ck, beam_size=args.beam_size, det_tta=args.det_tta, split=args.split,
                           aug_vocab=aug_vocab, orig_vocab=orig_vocab, aug_size=aug_size, orig_size=orig_size)
        rows.append(res)
        print(f"[{os.path.basename(ck)}] split={args.split} vocab={res['vocab_used']} "
              f"mIoU={res['mIoU']:.5f}, BLEU4={res['BLEU_4']:.5f}, time={res['time_sec']}s")

    fields = ['checkpoint', 'split', 'vocab_used', 'ck_vocab_size', 'ck_max_len',
              'beam_size', 'det_tta', 'time_sec',
              'Acc_seg', 'Acc_class_seg', 'mIoU', 'FWIoU',
              'BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L', 'CIDEr']
    with open(args.out_csv, 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f'Wrote CSV: {args.out_csv}')

if __name__ == '__main__':
    main()