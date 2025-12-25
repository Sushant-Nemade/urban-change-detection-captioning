
# predict_new.py

import os
import json
import argparse
import numpy as np
import torch
import cv2
from imageio.v2 import imread

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer

class Change_Perception(object):
    def define_args(self):
        p = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation — PREDICT')

        # Inputs
        p.add_argument('--imgA_path', required=True, help='path to A image')
        p.add_argument('--imgB_path', required=True, help='path to B image')
        p.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
        p.add_argument('--list_path', default='./data/LEVIR_MCI/', help='dir with vocab jsons')
        p.add_argument('--vocab_file', default='vocab', help='augmented vocab name (no .json)')
        p.add_argument('--orig_vocab_file', default='vocab_original', help='original vocab name (no .json)')

        # Outputs
        p.add_argument('--mask_save_path', default='./CDmask.png', help='where to save change mask PNG')
        p.add_argument('--save_mask_size', default='target', choices=['target', 'A', 'B'],
                       help='resize saved mask to 256x256 (target), imgA size, or imgB size')

        # Model config (match training)
        p.add_argument('--network', default='segformer-mit_b1')
        p.add_argument('--encoder_dim', type=int, default=512)
        p.add_argument('--feat_size', type=int, default=16)
        p.add_argument('--n_heads', type=int, default=8)
        p.add_argument('--n_layers', type=int, default=3)
        p.add_argument('--decoder_n_layers', type=int, default=1)
        p.add_argument('--feature_dim', type=int, default=512)
        p.add_argument('--dropout', type=float, default=0.1)

        # Inference options
        p.add_argument('--gpu_id', type=int, default=0)
        p.add_argument('--beam_size', type=int, default=5, help='1=greedy, >1=beam search (batch=1)')
        p.add_argument('--target_size', type=int, default=256, help='model input size (H=W)')

        return p.parse_args()

    def __init__(self):
        args = self.define_args()
        self.args = args
        torch.cuda.set_device(args.gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset normalization (training-time constants)
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std  = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        # Read checkpoint; infer decoder settings
        ckpt = torch.load(args.checkpoint, map_location=self.device)
        dec_sd = ckpt['decoder_dict']
        ck_vocab_size = dec_sd['wdc.weight'].shape[0]
        ck_max_len    = dec_sd['position_encoding.pe'].shape[0]

        # Choose vocab by checkpoint vocab size
        vocab_name = args.vocab_file if ck_vocab_size != 468 else args.orig_vocab_file
        vocab_path = os.path.join(args.list_path, vocab_name + '.json')

        self.caption_enabled = True
        if not os.path.exists(vocab_path):
            print(f'Warning: vocab file not found: {vocab_path}. Captions disabled; detection will still run.')
            self.caption_enabled = False
            self.word_vocab = {'<NULL>':0,'<UNK>':1,'<START>':2,'<END>':3}
        else:
            with open(vocab_path, 'r') as f:
                self.word_vocab = json.load(f)
            if len(self.word_vocab) != ck_vocab_size:
                raise RuntimeError(f'Vocab size mismatch: {vocab_name} has {len(self.word_vocab)} vs ckpt {ck_vocab_size}. '
                                   f'Use the correct vocab json.')

        self.idx_to_word = {v: k for k, v in self.word_vocab.items()}
        for s in ['<START>', '<END>', '<NULL>']:
            if s not in self.word_vocab:
                raise RuntimeError(f'Missing special token {s} in vocab: {vocab_name}')

        # Build model
        self.encoder = Encoder(args.network).eval().to(self.device)
        self.encoder_trans = AttentiveEncoder(
            train_stage=None, n_layers=args.n_layers,
            feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
            heads=args.n_heads, dropout=args.dropout
        ).eval().to(self.device)
        self.decoder = DecoderTransformer(
            encoder_dim=args.encoder_dim,
            feature_dim=args.feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=ck_max_len,          # from checkpoint
            word_vocab=self.word_vocab,
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout
        ).eval().to(self.device)

        # Load weights
        self.encoder.load_state_dict(ckpt['encoder_dict'], strict=True)
        self.encoder_trans.load_state_dict(ckpt['encoder_trans_dict'], strict=False)
        self.decoder.load_state_dict(dec_sd, strict=True)

    # ---------- Robust I/O and preprocessing ----------
    def _read_rgb3(self, path):
        """Read image as HWC float32 RGB (3 channels), handling grayscale/RGBA/extra bands."""
        arr = imread(path)
        if arr is None:
            raise FileNotFoundError(f'Cannot read image: {path}')
        # to numpy float32
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3:
            if arr.shape[2] == 4:         # RGBA -> RGB
                arr = arr[..., :3]
            elif arr.shape[2] > 4:        # more than 4 channels -> take first 3
                arr = arr[..., :3]
            elif arr.shape[2] == 1:       # single channel -> repeat
                arr = np.repeat(arr, 3, axis=-1)
            elif arr.shape[2] == 3:
                pass
            else:
                # unexpected, force 3
                arr = arr[..., :3] if arr.shape[2] >= 3 else np.repeat(arr, 3, axis=-1)
        else:
            raise ValueError(f'Unsupported image shape for {path}: {arr.shape}')
        return arr.astype(np.float32)

    def _resize_hwc(self, img, size):
        """Resize HWC image to square (size, size) with bilinear interpolation."""
        if img.shape[0] == size and img.shape[1] == size:
            return img
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    def _normalize_chw(self, img_hwc):
        """Normalize HWC to CHW with dataset mean/std."""
        chw = np.transpose(img_hwc, (2, 0, 1))
        for c in range(3):
            chw[c] = (chw[c] - self.mean[c]) / self.std[c]
        return chw

    def _prepare_images(self, path_A, path_B):
        """Read, force 3ch, record original sizes, resize to target, normalize, to torch."""
        t = int(self.args.target_size)
        imgA = self._read_rgb3(path_A)
        imgB = self._read_rgb3(path_B)
        origA = (imgA.shape[0], imgA.shape[1])  # (H,W)
        origB = (imgB.shape[0], imgB.shape[1])

        imgA = self._resize_hwc(imgA, t)
        imgB = self._resize_hwc(imgB, t)

        imgA = self._normalize_chw(imgA)
        imgB = self._normalize_chw(imgB)

        tenA = torch.from_numpy(imgA).float().unsqueeze(0).to(self.device, non_blocking=True)
        tenB = torch.from_numpy(imgB).float().unsqueeze(0).to(self.device, non_blocking=True)
        return tenA, tenB, origA, origB

    @torch.no_grad()
    def _forward_once(self, imgA, imgB):
        feat1_list, feat2_list = self.encoder(imgA, imgB)
        feat1_cap, feat2_cap, seg_pre = self.encoder_trans(feat1_list, feat2_list)
        return feat1_cap, feat2_cap, seg_pre

    # ---------- Caption ----------
    @torch.no_grad()
    def generate_change_caption(self, path_A, path_B):
        if not self.caption_enabled:
            print('Caption disabled (missing matching vocab for this checkpoint).')
            return ''
        imgA, imgB, _, _ = self._prepare_images(path_A, path_B)
        f1, f2, _ = self._forward_once(imgA, imgB)
        if self.args.beam_size > 1:
            seq = self.decoder.sample_beam(f1, f2, k=self.args.beam_size)
        else:
            seq = self.decoder.sample(f1, f2, k=1)
        specials = {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}
        if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
            seq = seq[0]
        tokens = [self.idx_to_word.get(i, '<UNK>') for i in seq if i not in specials]
        caption = ' '.join(tokens).strip()
        print('Caption:', caption)
        return caption

    # ---------- Change detection ----------
    @torch.no_grad()
    def change_detection(self, path_A, path_B, savepath_mask):
        imgA, imgB, origA, origB = self._prepare_images(path_A, path_B)
        _, _, seg_pre = self._forward_once(imgA, imgB)

        # Argmax over classes: 0=background, 1=road, 2=building
        pred = np.argmax(seg_pre.detach().cpu().numpy(), axis=1)[0].astype(np.uint8)

        # Optionally resize predicted label map to A or B original sizes for saving
        save_sz = self.args.save_mask_size
        if save_sz == 'A':
            target_hw = origA
        elif save_sz == 'B':
            target_hw = origB
        else:
            target_hw = (pred.shape[0], pred.shape[1])  # target (model) size

        if (pred.shape[0], pred.shape[1]) != target_hw:
            # Resize label map with nearest neighbor to keep classes intact
            pred = cv2.resize(pred, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

        # Colorize in RGB
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = (255, 255, 0)  # yellow for road
        pred_rgb[pred == 2] = (255, 0, 0)    # red for building

        os.makedirs(os.path.dirname(savepath_mask) or '.', exist_ok=True)
        # Convert RGB->BGR for OpenCV
        cv2.imwrite(savepath_mask, pred_rgb[:, :, ::-1])
        print('Mask saved:', savepath_mask)
        return pred

    # Optional: connected-component counting (roads/buildings)
    def compute_object_num(self, changed_mask, object_type):
        from skimage import measure
        mask = np.zeros_like(changed_mask, dtype=np.uint8)
        if object_type == 'road':
            mask[changed_mask == 1] = 1
        elif object_type == 'building':
            mask[changed_mask == 2] = 1
        else:
            raise ValueError("object_type must be 'road' or 'building'")
        lbl = measure.label(mask, connectivity=2)
        count = sum(1 for p in measure.regionprops(lbl) if p.area > 5)
        msg = f'Found {count} changed {object_type}'
        print(msg)
        return msg


if __name__ == '__main__':
    cp = Change_Perception()
    _ = cp.generate_change_caption(cp.args.imgA_path, cp.args.imgB_path)
    _ = cp.change_detection(cp.args.imgA_path, cp.args.imgB_path, cp.args.mask_save_path)