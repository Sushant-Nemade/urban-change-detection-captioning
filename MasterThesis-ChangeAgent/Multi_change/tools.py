import os
import json
import numpy as np
import torch
import cv2
from imageio.v2 import imread

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer


class Change_Perception:
    """
    Lightweight wrapper for model inference used by Change-Agent actions.

    Defaults:
    - checkpoint: ./models_ckpt/MCI_model_augmented.pth
    - list_path: ./data/LEVIR_MCI/
    - vocab auto-selects based on checkpoint (468 vs augmented)
    """

    def __init__(self,
                 checkpoint: str = './models_ckpt/MCI_model_augmented.pth',
                 list_path: str = './data/LEVIR_MCI/',
                 vocab_file: str = 'vocab',
                 orig_vocab_file: str = 'vocab_original',
                 network: str = 'segformer-mit_b1',
                 encoder_dim: int = 512,
                 feat_size: int = 16,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 decoder_n_layers: int = 1,
                 feature_dim: int = 512,
                 dropout: float = 0.1,
                 gpu_id: int = 0,
                 beam_size: int = 5):
        torch.cuda.set_device(gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.beam_size = beam_size

        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        ckpt = torch.load(checkpoint, map_location=self.device)
        dec_sd = ckpt['decoder_dict']
        ck_vocab_size = dec_sd['wdc.weight'].shape[0]
        ck_max_len = dec_sd['position_encoding.pe'].shape[0]

        vocab_name = vocab_file if ck_vocab_size != 468 else orig_vocab_file
        vocab_path = os.path.join(list_path, vocab_name + '.json')
        self.caption_enabled = True
        if not os.path.exists(vocab_path):
            self.caption_enabled = False
            self.word_vocab = {'<NULL>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        else:
            with open(vocab_path, 'r') as f:
                self.word_vocab = json.load(f)

        self.idx_to_word = {v: k for k, v in self.word_vocab.items()}

        self.encoder = Encoder(network).eval().to(self.device)
        self.encoder_trans = AttentiveEncoder(
            train_stage=None, n_layers=n_layers,
            feature_size=[feat_size, feat_size, encoder_dim],
            heads=n_heads, dropout=dropout
        ).eval().to(self.device)
        self.decoder = DecoderTransformer(
            encoder_dim=encoder_dim,
            feature_dim=feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=ck_max_len,
            word_vocab=self.word_vocab,
            n_head=n_heads,
            n_layers=decoder_n_layers,
            dropout=dropout
        ).eval().to(self.device)

        self.encoder.load_state_dict(ckpt['encoder_dict'], strict=True)
        self.encoder_trans.load_state_dict(ckpt['encoder_trans_dict'], strict=False)
        self.decoder.load_state_dict(dec_sd, strict=True)

    def _prepare_images(self, path_A, path_B):
        arrA = imread(path_A)
        arrB = imread(path_B)
        arrA = np.asarray(arrA)
        arrB = np.asarray(arrB)
        if arrA.ndim == 2:
            arrA = np.stack([arrA, arrA, arrA], axis=-1)
        if arrB.ndim == 2:
            arrB = np.stack([arrB, arrB, arrB], axis=-1)
        arrA = arrA.astype(np.float32)
        arrB = arrB.astype(np.float32)
        if (arrA.shape[0] != 256) or (arrA.shape[1] != 256):
            arrA = cv2.resize(arrA, (256, 256), interpolation=cv2.INTER_LINEAR)
        if (arrB.shape[0] != 256) or (arrB.shape[1] != 256):
            arrB = cv2.resize(arrB, (256, 256), interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            arrA[..., c] = (arrA[..., c] - self.mean[c]) / self.std[c]
            arrB[..., c] = (arrB[..., c] - self.mean[c]) / self.std[c]
        tenA = torch.from_numpy(np.transpose(arrA, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        tenB = torch.from_numpy(np.transpose(arrB, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        return tenA, tenB

    @torch.no_grad()
    def _forward_once(self, imgA, imgB):
        feat1_list, feat2_list = self.encoder(imgA, imgB)
        feat1_cap, feat2_cap, seg_pre = self.encoder_trans(feat1_list, feat2_list)
        return feat1_cap, feat2_cap, seg_pre

    @torch.no_grad()
    def generate_change_caption(self, path_A: str, path_B: str) -> str:
        if not self.caption_enabled:
            return ''
        imgA, imgB = self._prepare_images(path_A, path_B)
        f1, f2, _ = self._forward_once(imgA, imgB)
        if self.beam_size > 1:
            seq = self.decoder.sample_beam(f1, f2, k=self.beam_size)
        else:
            seq = self.decoder.sample(f1, f2, k=1)
        specials = {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}
        if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
            seq = seq[0]
        tokens = [self.idx_to_word.get(i, '<UNK>') for i in seq if i not in specials]
        return ' '.join(tokens).strip()

    @torch.no_grad()
    def change_detection(self, path_A: str, path_B: str, savepath_mask: str):
        imgA, imgB = self._prepare_images(path_A, path_B)
        _, _, seg_pre = self._forward_once(imgA, imgB)
        pred = np.argmax(seg_pre.detach().cpu().numpy(), axis=1)[0].astype(np.uint8)
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = (255, 255, 0)  # road -> yellow
        pred_rgb[pred == 2] = (255, 0, 0)    # building -> red
        os.makedirs(os.path.dirname(savepath_mask) or '.', exist_ok=True)
        cv2.imwrite(savepath_mask, pred_rgb[:, :, ::-1])
        return pred

    def compute_object_num(self, changed_mask, object: str):
        from skimage import measure
        mask = np.zeros_like(changed_mask, dtype=np.uint8)
        if object == 'road':
            mask[changed_mask == 1] = 1
        elif object == 'building':
            mask[changed_mask == 2] = 1
        else:
            raise ValueError("object must be 'road' or 'building'")
        lbl = measure.label(mask, connectivity=2)
        count = sum(1 for p in measure.regionprops(lbl) if p.area > 5)
        return count


if __name__ == '__main__':
    cp = Change_Perception()
    print('Change_Perception ready. Provide paths to A/B to run methods.')

