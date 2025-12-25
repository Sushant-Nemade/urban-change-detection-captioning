import os
import time
import cv2
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils import data

from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from utils_tool.metrics import Evaluator


def save_mask(pred, gt, name, save_path, args):
    # pred value: 0,1,2; map to black, yellow, red
    # gt value: 0,1,2; map to black, yellow, red
    name = name[0]
    evaluator = Evaluator(num_class=3)
    evaluator.add_batch(gt, pred)
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    Miou_str = round(mIoU_seg, 4)

    # write/update per-image score JSON
    json_path = os.path.join(save_path, 'score.json')
    key = name.split('.')[0]
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({key: {'MIoU': Miou_str}}, f)
    else:
        with open(json_path, 'r') as f:
            data = json.load(f)
        data[key] = {'MIoU': Miou_str}
        with open(json_path, 'w') as f:
            json.dump(data, f)

    # save predicted and GT masks as RGB for visualization
    pred = pred[0].astype(np.uint8)
    gt = gt[0].astype(np.uint8)
    pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    gt_rgb = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.uint8)
    pred_rgb[pred == 1] = [0, 255, 255]
    pred_rgb[pred == 2] = [0, 0, 255]
    gt_rgb[gt == 1] = [0, 255, 255]
    gt_rgb[gt == 2] = [0, 0, 255]
    cv2.imwrite(os.path.join(save_path, key + '_mask.png'), pred_rgb)
    cv2.imwrite(os.path.join(save_path, key + '_gt.png'), gt_rgb)

    # save image A/B for reference
    img_A_path = os.path.join(args.data_folder, 'test', 'A', name)
    img_B_path = os.path.join(args.data_folder, 'test', 'B', name)
    img_A = cv2.imread(img_A_path)
    img_B = cv2.imread(img_B_path)
    if img_A is not None:
        cv2.imwrite(os.path.join(save_path, key + '_A.png'), img_A)
    if img_B is not None:
        cv2.imwrite(os.path.join(save_path, key + '_B.png'), img_B)


def save_captions(pred_caption, ref_caption_all, hypotheses, references, name, save_path):
    # Store per-image BLEU-3/4 and strings
    name = name[0]
    key = name.split('.')[0]

    score_dict = get_eval_score([references], [hypotheses])
    Bleu_4 = round(score_dict['Bleu_4'], 4)
    Bleu_3 = round(score_dict['Bleu_3'], 4)

    json_path = os.path.join(save_path, 'score.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}
    data.setdefault(key, {})
    data[key]['Bleu_3'] = Bleu_3
    data[key]['Bleu_4'] = Bleu_4
    with open(json_path, 'w') as f:
        json.dump(data, f)

    with open(os.path.join(save_path, key + '_cap.txt'), 'w') as f:
        f.write('pred_caption: ' + pred_caption + '\n')
        f.write('ref_captions: ' + ref_caption_all + '\n')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab (token -> idx) and build inverse (idx -> token) for safe decoding
    with open(os.path.join(args.list_path, args.vocab_file + '.json'), 'r') as f:
        word_vocab = json.load(f)
    idx_to_word = {v: k for k, v in word_vocab.items()}

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Prepare result folder
    args.result_path = os.path.join(args.result_path, os.path.basename(args.checkpoint).replace('.pth', ''))
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    else:
        print('result_path exists, clearing...')
        for root, dirs, files in os.walk(args.result_path):
            for fn in files:
                try:
                    os.remove(os.path.join(root, fn))
                except Exception:
                    pass
            for dn in dirs:
                try:
                    os.rmdir(os.path.join(root, dn))
                except Exception:
                    pass

    # Build models
    encoder = Encoder(args.network)
    encoder_trans = AttentiveEncoder(
        train_stage=None, n_layers=args.n_layers,
        feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
        heads=args.n_heads, dropout=args.dropout
    )
    decoder = DecoderTransformer(
        encoder_dim=args.encoder_dim,
        feature_dim=args.feature_dim,
        vocab_size=len(word_vocab),
        max_lengths=args.max_length,
        word_vocab=word_vocab,
        n_head=args.n_heads,
        n_layers=args.decoder_n_layers,
        dropout=args.dropout
    )

    # Load weights
    encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
    encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
    decoder.load_state_dict(checkpoint['decoder_dict'], strict=True)

    # To device
    encoder.eval().to(device)
    encoder_trans.eval().to(device)
    decoder.eval().to(device)

    # Data
    if args.data_name == 'LEVIR_MCI':
        test_loader = data.DataLoader(
            LEVIRCCDataset(args.data_folder, args.list_path, 'test',
                           args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
            batch_size=args.test_batchsize, shuffle=False, num_workers=args.workers, pin_memory=True
        )
    else:
        raise ValueError('Unsupported data_name')

    # Eval containers
    evaluator = Evaluator(num_class=3)
    references, hypotheses = [], []
    change_references, change_hypotheses = [], []
    nochange_references, nochange_hypotheses = [], []
    change_acc = 0
    nochange_acc = 0

    nochange_list = {
        "the scene is the same as before ",
        "there is no difference ",
        "the two scenes seem identical ",
        "no change has occurred ",
        "almost nothing has changed "
    }

    def forward_once(imgA, imgB):
        feat1_list, feat2_list = encoder(imgA, imgB)
        feat1, feat2, seg_pre = encoder_trans(feat1_list, feat2_list)
        return feat1, feat2, seg_pre

    test_start_time = time.time()
    with torch.no_grad():
        for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, name) in enumerate(
                tqdm(test_loader, desc=f'test_BEAM {args.beam_size}')):
            imgA = imgA.to(device, non_blocking=True)
            imgB = imgB.to(device, non_blocking=True)
            token_all = token_all.squeeze(0).to(device, non_blocking=True)  # 5 refs

            # Forward: get caption features and base segmentation
            feat1, feat2, seg_pre = forward_once(imgA, imgB)

            # Optional detection TTA (H/V flips on seg only)
            if args.det_tta == 1:
                segs = [seg_pre]
                # H flip (height dim = -2)
                _, _, seg_h = forward_once(torch.flip(imgA, dims=[-2]), torch.flip(imgB, dims=[-2]))
                segs.append(torch.flip(seg_h, dims=[-2]))
                # V flip (width dim = -1)
                _, _, seg_v = forward_once(torch.flip(imgA, dims=[-1]), torch.flip(imgB, dims=[-1]))
                segs.append(torch.flip(seg_v, dims=[-1]))
                seg_pre = torch.mean(torch.stack(segs, dim=0), dim=0)

            # Decode caption
            if args.beam_size > 1:
                seq = decoder.sample_beam(feat1, feat2, k=args.beam_size)
            else:
                seq = decoder.sample(feat1, feat2, k=1)

            # Segmentation metrics
            pred_seg = seg_pre.data.cpu().numpy()
            seg_label_np = seg_label.numpy()
            pred_seg = np.argmax(pred_seg, axis=1)
            evaluator.add_batch(seg_label_np, pred_seg)

            # Optionally save visuals
            if args.save_mask:
                save_mask(pred_seg, seg_label_np, name, args.result_path, args)

            # Caption references and hypothesis (remove special tokens)
            specials = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
            img_token = token_all.tolist()
            img_tokens = list(map(lambda c: [w for w in c if w not in specials], img_token))
            references.append(img_tokens)

            if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
                # just in case beam returns nested (shouldn't with our code), take first
                seq = seq[0]
            pred_seq = [w for w in seq if w not in specials]
            hypotheses.append(pred_seq)
            assert len(references) == len(hypotheses)

            # Strings for optional saving and nochange split
            pred_caption = " ".join(idx_to_word.get(i, '<UNK>') for i in pred_seq) + " "
            ref_captions_all = ""
            for ref in img_tokens:
                ref_captions_all += " ".join(idx_to_word.get(i, '<UNK>') for i in ref) + ".    "

            if args.save_caption:
                save_captions(pred_caption, ref_captions_all, hypotheses[-1], references[-1], name, args.result_path)

            # no-change vs change split (rough heuristic)
            first_ref = " ".join(idx_to_word.get(i, '<UNK>') for i in img_tokens[0]) + " "
            if first_ref in nochange_list:
                nochange_references.append(img_tokens)
                nochange_hypotheses.append(pred_seq)
                if pred_caption in nochange_list:
                    nochange_acc += 1
            else:
                change_references.append(img_tokens)
                change_hypotheses.append(pred_seq)
                if pred_caption not in nochange_list:
                    change_acc += 1

    test_time = time.time() - test_start_time

    # Report detection
    Acc_seg = evaluator.Pixel_Accuracy()
    Acc_class_seg = evaluator.Pixel_Accuracy_Class()
    mIoU_seg, IoU = evaluator.Mean_Intersection_over_Union()
    FWIoU_seg = evaluator.Frequency_Weighted_Intersection_over_Union()
    print('Detection (TEST):')
    print(f'Acc_seg: {Acc_seg:.5f}  Acc_class_seg: {Acc_class_seg:.5f}  mIoU_seg: {mIoU_seg:.5f}  FWIoU_seg: {FWIoU_seg:.5f}')
    print('IoU:', IoU)

    # Split metrics
    if len(nochange_references) > 0:
        print('nochange_metric:')
        nochange_metric = get_eval_score(nochange_references, nochange_hypotheses)
        print('BLEU-1: {0:.5f}  BLEU-2: {1:.5f}  BLEU-3: {2:.5f}  BLEU-4: {3:.5f}  Meteor: {4:.5f}  Rouge: {5:.5f}  Cider: {6:.5f}'
              .format(nochange_metric['Bleu_1'], nochange_metric['Bleu_2'], nochange_metric['Bleu_3'],
                      nochange_metric['Bleu_4'], nochange_metric['METEOR'],
                      nochange_metric['ROUGE_L'], nochange_metric['CIDEr']))
        print('nochange_acc:', nochange_acc / len(nochange_references))
    if len(change_references) > 0:
        print('change_metric:')
        change_metric = get_eval_score(change_references, change_hypotheses)
        print('BLEU-1: {0:.5f}  BLEU-2: {1:.5f}  BLEU-3: {2:.5f}  BLEU-4: {3:.5f}  Meteor: {4:.5f}  Rouge: {5:.5f}  Cider: {6:.5f}'
              .format(change_metric['Bleu_1'], change_metric['Bleu_2'], change_metric['Bleu_3'],
                      change_metric['Bleu_4'], change_metric['METEOR'],
                      change_metric['ROUGE_L'], change_metric['CIDEr']))
        print('change_acc:', change_acc / len(change_references))

    # Overall captioning
    score_dict = get_eval_score(references, hypotheses)
    print('Captioning (TEST):')
    print('Time: {0:.3f}  BLEU-1: {1:.5f}  BLEU-2: {2:.5f}  BLEU-3: {3:.5f}  BLEU-4: {4:.5f}  Meteor: {5:.5f}  Rouge: {6:.5f}  Cider: {7:.5f}'
          .format(test_time, score_dict['Bleu_1'], score_dict['Bleu_2'], score_dict['Bleu_3'],
                  score_dict['Bleu_4'], score_dict['METEOR'], score_dict['ROUGE_L'], score_dict['CIDEr']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation — TEST')

    # Data parameters
    parser.add_argument('--sys', default='win', help='system win or linux')
    parser.add_argument('--data_folder', default=r'D:\Dataset\Caption\change_caption\Levir-MCI-dataset\images',
                        help='folder with image files')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='path of the data lists')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/', help='folder with token files')
    parser.add_argument('--vocab_file', default='vocab', help='vocab json (without .json extension)')
    parser.add_argument('--max_length', type=int, default=46, help='max caption length (match training)')
    parser.add_argument('--allow_unk', type=int, default=1, help='if unknown token is allowed')
    parser.add_argument('--data_name', default='LEVIR_MCI', help='dataset name')

    # Test
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth', help='path to checkpoint')
    parser.add_argument('--print_freq', type=int, default=100, help='unused (kept for compatibility)')
    parser.add_argument('--test_batchsize', type=int, default=1, help='batch size for test (beam>1 requires 1)')
    parser.add_argument('--workers', type=int, default=0, help='for data-loading')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--beam_size', type=int, default=5, help='beam size for caption decoding (1=greedy)')
    parser.add_argument('--det_tta', type=int, default=0, help='if 1: average seg logits over H/V flips')

    # Save outputs
    parser.add_argument('--save_mask', action='store_true', help='save predicted masks and GT')
    parser.add_argument('--save_caption', action='store_true', help='save predicted and reference captions')
    parser.add_argument('--result_path', default='./predict_result/', help='where to save masks/captions')

    # Encoder/neck/decoder params (must match training config)
    parser.add_argument('--network', default='segformer-mit_b1', help='backbone encoder')
    parser.add_argument('--encoder_dim', type=int, default=512, help='last feature dim')
    parser.add_argument('--feat_size', type=int, default=16, help='feature spatial size (256/16=16)')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3, help='AttentiveEncoder layers')
    parser.add_argument('--decoder_n_layers', type=int, default=1, help='decoder TF layers')
    parser.add_argument('--feature_dim', type=int, default=512, help='decoder embedding dim')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    main(args)