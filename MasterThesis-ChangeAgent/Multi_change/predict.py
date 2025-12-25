# # import sys
# #
# # # 添加特定路径到 Python 解释器的搜索路径中
# # # sys.path.append('F:\LCY\Change_Agent\Change-Agent-git\Multi_change')
# # import os.path
# #
# # import cv2
# # import torch.optim
# # import argparse
# # import json
# #
# # from skimage import measure
# #
# # from model.model_encoder_att import Encoder, AttentiveEncoder
# # from model.model_decoder import DecoderTransformer
# # from utils_tool.utils import *
# # from imageio.v2 import imread
# #
# #
# # # compute_change_map(path_A, path_B)函数: 生成一个掩膜mask用来表示两个图像之间的变化区域
# # '''
# # Args:
# #     path_A: 图像A的路径
# #     path_B: 图像B的路径
# # Returns:
# #     change_map: 变化区域的掩膜
# # '''
# # # def compute_change_mask(path_A, path_B):
# # #     import cv2
# # #     import numpy as np
# # #     img_A = cv2.imread(path_A)
# # #     img_B = cv2.imread(path_B)
# # #     change_map = (img_B-img_A).astype(np.uint8)
# # #     # 阈值化
# # #     change_map = cv2.cvtColor(change_map, cv2.COLOR_BGR2GRAY)
# # #     change_map = cv2.threshold(change_map, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# # #     cv2.imwrite('E:\change_map.png', change_map)
# # #     return 'I have save the changed mask in E:\change_map.png'
# #
# # # compute_change_caption(path_A, path_B)函数：生成一个文本用于描述两个图像之间变化
# # '''
# # Args:
# #     path_A: 图像A的路径
# #     path_B: 图像B的路径
# # Returns:
# #     caption: 变化描述文本
# # '''
# # class Change_Perception(object):
# #     def define_args(self):
# #
# #
# #         script_path = os.path.abspath(__file__)
# #         script_dir = os.path.dirname(script_path)
# #         print(script_dir)
# #         parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
# #
# #         parser.add_argument('--data_folder', default='D:\Dataset\Caption\change_caption\Levir-MCI-dataset\images',
# #                             help='folder with data files')
# #         parser.add_argument('--list_path', default='F:\LCY\Change_Agent\Change-Agent-git\Multi_change\data\LEVIR_MCI/',
# #                             help='path of the data lists')
# #         parser.add_argument('--vocab_file', default='vocab', help='path of the data lists')
# #         parser.add_argument('--max_length', type=int, default=41, help='path of the data lists')
# #
# #         # inference
# #         parser.add_argument('--gpu_id', type=int, default=0, help='gpu id in the training.')
# #         parser.add_argument('--checkpoint', default='./models_ckpt/MCI_model.pth',help='path to checkpoint')
# #         parser.add_argument('--result_path', default="./predict_result/",
# #                             help='path to save the result of masks and captions')
# #
# #         # backbone parameters
# #         parser.add_argument('--network', default='segformer-mit_b1',
# #                             help='define the backbone encoder to extract features')
# #         parser.add_argument('--encoder_dim', type=int, default=512,
# #                             help='the dimension of extracted features using backbone ')
# #         parser.add_argument('--feat_size', type=int, default=16,
# #                             help='define the output size of encoder to extract features')
# #         parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
# #         # Model parameters
# #         parser.add_argument('--n_heads', type=int, default=8, help='Multi-head attention in Transformer.')
# #         parser.add_argument('--n_layers', type=int, default=3, help='Number of layers in AttentionEncoder.')
# #         parser.add_argument('--decoder_n_layers', type=int, default=1)
# #         parser.add_argument('--feature_dim', type=int, default=512, help='embedding dimension')
# #
# #         args = parser.parse_args()
# #
# #         return args
# #
# #     def __init__(self,):
# #         """
# #         Training and validation.
# #         """
# #         args = self.define_args()
# #         self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
# #         self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
# #
# #         with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
# #             self.word_vocab = json.load(f)
# #         # Load checkpoint
# #         snapshot_full_path = args.checkpoint
# #
# #         checkpoint = torch.load(snapshot_full_path)
# #         self.encoder = Encoder(args.network)
# #         self.encoder_trans = AttentiveEncoder(train_stage=None, n_layers=args.n_layers,
# #                                          feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
# #                                          heads=args.n_heads, dropout=args.dropout)
# #         self.decoder = DecoderTransformer(encoder_dim=args.encoder_dim, feature_dim=args.feature_dim,
# #                                      vocab_size=len(self.word_vocab), max_lengths=args.max_length,
# #                                      word_vocab=self.word_vocab, n_head=args.n_heads,
# #                                      n_layers=args.decoder_n_layers, dropout=args.dropout)
# #
# #         self.encoder.load_state_dict(checkpoint['encoder_dict'])
# #         self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
# #         self.decoder.load_state_dict(checkpoint['decoder_dict'])
# #         # Move to GPU, if available
# #         self.encoder.eval()
# #         self.encoder = self.encoder.cuda()
# #         self.encoder_trans.eval()
# #         self.encoder_trans = self.encoder_trans.cuda()
# #         self.decoder.eval()
# #         self.decoder = self.decoder.cuda()
# #
# #
# #     def preprocess(self, path_A, path_B):
# #
# #         imgA = imread(path_A)
# #         imgB = imread(path_B)
# #         imgA = np.asarray(imgA, np.float32)
# #         imgB = np.asarray(imgB, np.float32)
# #
# #         imgA = imgA.transpose(2, 0, 1)
# #         imgB = imgB.transpose(2, 0, 1)
# #         for i in range(len(self.mean)):
# #             imgA[i, :, :] -= self.mean[i]
# #             imgA[i, :, :] /= self.std[i]
# #             imgB[i, :, :] -= self.mean[i]
# #             imgB[i, :, :] /= self.std[i]
# #
# #         if imgA.shape[1] != 256 or imgA.shape[2] != 256:
# #             imgA = cv2.resize(imgA, (256, 256))
# #             imgB = cv2.resize(imgB, (256, 256))
# #
# #         imgA = torch.FloatTensor(imgA)
# #         imgB = torch.FloatTensor(imgB)
# #         imgA = imgA.unsqueeze(0)  # (1, 3, 256, 256)
# #         imgB = imgB.unsqueeze(0)
# #
# #         return imgA, imgB
# #
# #     def generate_change_caption(self, path_A, path_B):
# #         print('model_infer_change_captioning: start')
# #         imgA, imgB = self.preprocess(path_A, path_B)
# #         # Move to GPU, if available
# #         imgA = imgA.cuda()
# #         imgB = imgB.cuda()
# #         feat1, feat2 = self.encoder(imgA, imgB)
# #         feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
# #         seq = self.decoder.sample(feat1, feat2, k=1)
# #         pred_seq = [w for w in seq if w not in {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}]
# #         pred_caption = ""
# #         for i in pred_seq:
# #             pred_caption += (list(self.word_vocab.keys())[i]) + " "
# #
# #         caption ='there is road change'
# #         caption = pred_caption
# #         print('change captioning:', caption)
# #         return caption
# #
# #     def change_detection(self, path_A, path_B, savepath_mask):
# #         print('model_infer_change_detection: start')
# #         imgA, imgB = self.preprocess(path_A, path_B)
# #         # Move to GPU, if available
# #         imgA = imgA.cuda()
# #         imgB = imgB.cuda()
# #         feat1, feat2 = self.encoder(imgA, imgB)
# #         feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
# #         # for segmentation
# #         pred_seg = seg_pre.data.cpu().numpy()
# #         pred_seg = np.argmax(pred_seg, axis=1)
# #         # 保存图片
# #         pred = pred_seg[0].astype(np.uint8)
# #         pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
# #         pred_rgb[pred == 1] = [0, 255, 255]
# #         pred_rgb[pred == 2] = [0, 0, 255]
# #
# #         cv2.imwrite(savepath_mask, pred_rgb)
# #         print('model_infer: mask saved in', savepath_mask)
# #
# #         print('model_infer_change_detection: end')
# #         return pred # (256,256,3)
# #         # return 'change detection successfully. '
# #
# #     def compute_object_num(self, changed_mask, object):
# #         print("compute num start")
# #         # compute the number of connected components
# #         mask = changed_mask
# #         mask_cp = 0 * mask.copy()
# #         if object == 'road':
# #             mask_cp[mask == 1] = 255
# #         elif object == 'building':
# #             mask_cp[mask == 2] = 255
# #         lbl = measure.label(mask_cp, connectivity=2)
# #         props = measure.regionprops(lbl)
# #         # get bboxes by a for loop
# #         bboxes = []
# #         for prop in props:
# #             # print('Found bbox', prop.bbox, 'area:', prop.area)
# #             if prop.area > 5:
# #                 bboxes.append([prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]])
# #         num = len(bboxes)
# #         # visual
# #         # mask_array_copy = mask.copy()*255
# #         # for bbox in bboxes:
# #         #     print('Found bbox', bbox)
# #         #     cv2.rectangle(mask_array_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), 2)
# #         # cv2.namedWindow('findCorners', 0)
# #         # cv2.resizeWindow('findCorners', 600, 600)
# #         # cv2.imshow('findCorners', mask_array_copy)
# #         # cv2.waitKey(0)
# #         print('Found', num, object)
# #         print('compute num end')
# #         # return
# #         num_str = 'Found ' + str(num) + ' changed ' + object
# #         return num_str
# #
# #     # design more tool functions:
# #
# #
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
# #     parser.add_argument('--imgA_path', default=r'F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_A.png')
# #     parser.add_argument('--imgB_path', default=r'F:/LCY/Change_Agent/Multi_change/predict_result/test_000004_B.png')
# #     parser.add_argument('--mask_save_path', default=r'./CDmask.png')
# #
# #     args = parser.parse_args()
# #
# #     imgA_path = args.imgA_path
# #     imgB_path = args.imgB_path
# #
# #     Change_Perception = Change_Perception()
# #     Change_Perception.generate_change_caption(imgA_path, imgB_path)
# #     Change_Perception.change_detection(imgA_path, imgB_path, args.mask_save_path)
# #
# #
# # ############################## NEW Predict code given below #######################
#
#
# # predict.py
#
# import os
# import json
# import argparse
# import numpy as np
# import torch
# import cv2
# from imageio.v2 import imread
#
# from model.model_encoder_att import Encoder, AttentiveEncoder
# from model.model_decoder import DecoderTransformer
#
# class Change_Perception(object):
#     def define_args(self):
#         parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation — PREDICT')
#
#         # Inputs
#         parser.add_argument('--imgA_path', required=True, help='path to A image')
#         parser.add_argument('--imgB_path', required=True, help='path to B image')
#         parser.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
#         parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='data list dir (to load vocab)')
#         parser.add_argument('--vocab_file', default='vocab', help='vocab json filename without .json')
#
#         # Outputs
#         parser.add_argument('--mask_save_path', default='./CDmask.png', help='where to save change mask PNG')
#
#         # Model config (must match training)
#         parser.add_argument('--network', default='segformer-mit_b1', help='encoder backbone')
#         parser.add_argument('--encoder_dim', type=int, default=512)
#         parser.add_argument('--feat_size', type=int, default=16)
#         parser.add_argument('--n_heads', type=int, default=8)
#         parser.add_argument('--n_layers', type=int, default=3)
#         parser.add_argument('--decoder_n_layers', type=int, default=1)
#         parser.add_argument('--feature_dim', type=int, default=512)
#         parser.add_argument('--dropout', type=float, default=0.1)
#
#         # Inference options
#         parser.add_argument('--gpu_id', type=int, default=0)
#         parser.add_argument('--beam_size', type=int, default=5, help='1=greedy; >1 = beam search (batch=1)')
#
#         return parser.parse_args()
#
#     def __init__(self):
#         args = self.define_args()
#         self.args = args
#         torch.cuda.set_device(args.gpu_id)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         # Dataset mean/std (exactly as in training)
#         self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
#         self.std  = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
#
#         # Load vocab and idx->word map
#         vocab_path = os.path.join(args.list_path, args.vocab_file + '.json')
#         with open(vocab_path, 'r') as f:
#             self.word_vocab = json.load(f)
#         self.idx_to_word = {v: k for k, v in self.word_vocab.items()}
#         specials = ['<START>', '<END>', '<NULL>']
#         for s in specials:
#             if s not in self.word_vocab:
#                 raise RuntimeError(f'Missing special token {s} in vocab.')
#
#         # Read checkpoint and get decoder settings from it (robust)
#         ckpt = torch.load(args.checkpoint, map_location=self.device)
#         dec_sd = ckpt['decoder_dict']
#         ck_vocab_size = dec_sd['wdc.weight'].shape[0]
#         ck_max_len    = dec_sd['position_encoding.pe'].shape[0]
#
#         if len(self.word_vocab) != ck_vocab_size:
#             raise RuntimeError(f'Vocab size mismatch: vocab json={len(self.word_vocab)} vs ckpt={ck_vocab_size}. '
#                                f'Load the correct vocab for this checkpoint.')
#
#         # Build models
#         self.encoder = Encoder(args.network).eval().to(self.device)
#         self.encoder_trans = AttentiveEncoder(
#             train_stage=None,
#             n_layers=args.n_layers,
#             feature_size=[args.feat_size, args.feat_size, args.encoder_dim],
#             heads=args.n_heads,
#             dropout=args.dropout
#         ).eval().to(self.device)
#         self.decoder = DecoderTransformer(
#             encoder_dim=args.encoder_dim,
#             feature_dim=args.feature_dim,
#             vocab_size=len(self.word_vocab),
#             max_lengths=ck_max_len,         # from checkpoint
#             word_vocab=self.word_vocab,
#             n_head=args.n_heads,
#             n_layers=args.decoder_n_layers,
#             dropout=args.dropout
#         ).eval().to(self.device)
#
#         # Load weights
#         self.encoder.load_state_dict(ckpt['encoder_dict'], strict=True)
#         self.encoder_trans.load_state_dict(ckpt['encoder_trans_dict'], strict=False)
#         self.decoder.load_state_dict(dec_sd, strict=True)
#
#     def _read_and_preprocess(self, path_A, path_B):
#         # Read as RGB, float32
#         imgA = imread(path_A)
#         imgB = imread(path_B)
#         if imgA.ndim == 2: imgA = np.stack([imgA, imgA, imgA], axis=-1)
#         if imgB.ndim == 2: imgB = np.stack([imgB, imgB, imgB], axis=-1)
#         imgA = imgA.astype(np.float32)
#         imgB = imgB.astype(np.float32)
#
#         # Resize to 256x256 on HWC (same as dataset)
#         if (imgA.shape[0] != 256) or (imgA.shape[1] != 256):
#             imgA = cv2.resize(imgA, (256, 256), interpolation=cv2.INTER_LINEAR)
#         if (imgB.shape[0] != 256) or (imgB.shape[1] != 256):
#             imgB = cv2.resize(imgB, (256, 256), interpolation=cv2.INTER_LINEAR)
#
#         # Normalize (HWC) then to CHW
#         for c in range(3):
#             imgA[..., c] = (imgA[..., c] - self.mean[c]) / self.std[c]
#             imgB[..., c] = (imgB[..., c] - self.mean[c]) / self.std[c]
#         imgA = np.transpose(imgA, (2, 0, 1))
#         imgB = np.transpose(imgB, (2, 0, 1))
#
#         # To torch (batch=1)
#         imgA = torch.from_numpy(imgA).float().unsqueeze(0).to(self.device, non_blocking=True)
#         imgB = torch.from_numpy(imgB).float().unsqueeze(0).to(self.device, non_blocking=True)
#         return imgA, imgB
#
#     @torch.no_grad()
#     def _forward_once(self, imgA, imgB):
#         feat1_list, feat2_list = self.encoder(imgA, imgB)
#         feat1_cap, feat2_cap, seg_pre = self.encoder_trans(feat1_list, feat2_list)
#         return feat1_cap, feat2_cap, seg_pre
#
#     @torch.no_grad()
#     def generate_change_caption(self, path_A, path_B):
#         imgA, imgB = self._read_and_preprocess(path_A, path_B)
#         f1, f2, _ = self._forward_once(imgA, imgB)
#         if self.args.beam_size > 1:
#             seq = self.decoder.sample_beam(f1, f2, k=self.args.beam_size)
#         else:
#             seq = self.decoder.sample(f1, f2, k=1)
#
#         # Remove special tokens and decode
#         specials = {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}
#         if isinstance(seq, list) and (len(seq) > 0) and isinstance(seq[0], list):
#             seq = seq[0]
#         tokens = [self.idx_to_word.get(i, '<UNK>') for i in seq if i not in specials]
#         caption = ' '.join(tokens).strip()
#         print('Caption:', caption)
#         return caption
#
#     @torch.no_grad()
#     # def change_detection(self, path_A, path_B, savepath_mask):
#     #     imgA, imgB = self._read_and_preprocess(path_A, path_B)
#     #     _, _, seg_pre = self._forward_once(imgA, imgB)
#     #
#     #     # Argmax over classes -> 0/1/2
#     #     pred_seg = seg_pre.detach().cpu().numpy()
#     #     pred_seg = np.argmax(pred_seg, axis=1)  # (1,H,W)
#     #     pred = pred_seg[0].astype(np.uint8)
#     #
#     #     # Colorize: 1 -> yellow (road), 2 -> red (building), 0 -> black
#     #     pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#     #     pred_rgb[pred == 1] = [0, 255, 255]
#     #     pred_rgb[pred == 2] = [0, 0, 255]
#     #     os.makedirs(os.path.dirname(savepath_mask) or '.', exist_ok=True)
#     #     cv2.imwrite(savepath_mask, pred_rgb[:, :, ::-1])  # cv2 expects BGR
#     #     print('Mask saved:', savepath_mask)
#     #     return pred  # (H,W) with values in {0,1,2}
#
#     def change_detection(self, path_A, path_B, savepath_mask):
#         print('model_infer_change_detection: start')
#         imgA, imgB = self._read_and_preprocess(path_A, path_B)
#         imgA = imgA.cuda();
#         imgB = imgB.cuda()
#
#         feat1, feat2 = self.encoder(imgA, imgB)
#         feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
#
#         # Argmax over classes: 0=background, 1=road, 2=building
#         pred_seg = seg_pre.data.cpu().numpy()
#         pred = np.argmax(pred_seg, axis=1)[0].astype(np.uint8)
#
#         # Colorize in RGB
#         pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
#         # road -> yellow (RGB = 255,255,0)
#         pred_rgb[pred == 1] = (255, 255, 0)
#         # building -> red (RGB = 255,0,0)
#         pred_rgb[pred == 2] = (255, 0, 0)
#
#         # Save with OpenCV (expects BGR); convert RGB->BGR
#         cv2.imwrite(savepath_mask, pred_rgb[:, :, ::-1])
#         print('model_infer: mask saved in', savepath_mask)
#         print('model_infer_change_detection: end')
#         return pred
#
#     def compute_object_num(self, changed_mask, object_type):
#         # Count connected components of a given class in the 2D label mask
#         from skimage import measure
#         mask = np.zeros_like(changed_mask, dtype=np.uint8)
#         if object_type == 'road':
#             mask[changed_mask == 1] = 1
#         elif object_type == 'building':
#             mask[changed_mask == 2] = 1
#         else:
#             raise ValueError("object_type must be 'road' or 'building'")
#         lbl = measure.label(mask, connectivity=2)
#         # Filter tiny regions
#         count = sum(1 for p in measure.regionprops(lbl) if p.area > 5)
#         msg = f'Found {count} changed {object_type}'
#         print(msg)
#         return msg
#
# if __name__ == '__main__':
#     cp = Change_Perception()
#     cap = cp.generate_change_caption(cp.args.imgA_path, cp.args.imgB_path)
#     mask = cp.change_detection(cp.args.imgA_path, cp.args.imgB_path, cp.args.mask_save_path)


######################################################################################################################



# predict.py

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
        parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation — PREDICT')

        # Inputs
        parser.add_argument('--imgA_path', required=True, help='path to A image')
        parser.add_argument('--imgB_path', required=True, help='path to B image')
        parser.add_argument('--checkpoint', required=True, help='path to .pth checkpoint')
        parser.add_argument('--list_path', default='./data/LEVIR_MCI/', help='data list dir')
        parser.add_argument('--vocab_file', default='vocab', help='augmented vocab json filename (no .json)')
        parser.add_argument('--orig_vocab_file', default='vocab_original', help='original vocab json (no .json)')

        # Outputs
        parser.add_argument('--mask_save_path', default='./CDmask.png', help='save change mask here')

        # Model config (match training)
        parser.add_argument('--network', default='segformer-mit_b1')
        parser.add_argument('--encoder_dim', type=int, default=512)
        parser.add_argument('--feat_size', type=int, default=16)
        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--decoder_n_layers', type=int, default=1)
        parser.add_argument('--feature_dim', type=int, default=512)
        parser.add_argument('--dropout', type=float, default=0.1)

        # Inference
        parser.add_argument('--gpu_id', type=int, default=0)
        parser.add_argument('--beam_size', type=int, default=5)

        return parser.parse_args()

    def __init__(self):
        args = self.define_args()
        self.args = args
        torch.cuda.set_device(args.gpu_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataset normalization
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std  = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        # Read checkpoint and infer decoder settings
        ckpt = torch.load(args.checkpoint, map_location=self.device)
        dec_sd = ckpt['decoder_dict']
        ck_vocab_size = dec_sd['wdc.weight'].shape[0]
        ck_max_len    = dec_sd['position_encoding.pe'].shape[0]

        # Pick vocab based on ckpt vocab_size
        vocab_name = args.vocab_file
        if ck_vocab_size == 468:
            vocab_name = args.orig_vocab_file  # original MCI
        vocab_path = os.path.join(args.list_path, vocab_name + '.json')

        self.caption_enabled = True
        if not os.path.exists(vocab_path):
            print(f'Warning: vocab file not found: {vocab_path}. Captions disabled; detection will still run.')
            self.caption_enabled = False
            # Minimal specials for decoder to run; but no readable text
            self.word_vocab = {'<NULL>':0,'<UNK>':1,'<START>':2,'<END>':3}
        else:
            with open(vocab_path, 'r') as f:
                self.word_vocab = json.load(f)
            if len(self.word_vocab) != ck_vocab_size:
                raise RuntimeError(f'Vocab size mismatch: {vocab_name} has {len(self.word_vocab)} vs ckpt {ck_vocab_size}. '
                                   f'Use the correct vocab for this checkpoint.')

        self.idx_to_word = {v: k for k, v in self.word_vocab.items()}
        specials = ['<START>', '<END>', '<NULL>']
        for s in specials:
            if s not in self.word_vocab:
                raise RuntimeError(f'Missing special token {s} in vocab: {vocab_name}')

        # Build models
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

    def _read_and_preprocess(self, path_A, path_B):
        imgA = imread(path_A); imgB = imread(path_B)
        if imgA.ndim == 2: imgA = np.stack([imgA, imgA, imgA], axis=-1)
        if imgB.ndim == 2: imgB = np.stack([imgB, imgB, imgB], axis=-1)
        imgA = imgA.astype(np.float32); imgB = imgB.astype(np.float32)
        if (imgA.shape[0] != 256) or (imgA.shape[1] != 256):
            imgA = cv2.resize(imgA, (256, 256), interpolation=cv2.INTER_LINEAR)
        if (imgB.shape[0] != 256) or (imgB.shape[1] != 256):
            imgB = cv2.resize(imgB, (256, 256), interpolation=cv2.INTER_LINEAR)
        for c in range(3):
            imgA[..., c] = (imgA[..., c] - self.mean[c]) / self.std[c]
            imgB[..., c] = (imgB[..., c] - self.mean[c]) / self.std[c]
        imgA = torch.from_numpy(np.transpose(imgA, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        imgB = torch.from_numpy(np.transpose(imgB, (2, 0, 1))).float().unsqueeze(0).to(self.device)
        return imgA, imgB

    @torch.no_grad()
    def _forward_once(self, imgA, imgB):
        feat1_list, feat2_list = self.encoder(imgA, imgB)
        feat1_cap, feat2_cap, seg_pre = self.encoder_trans(feat1_list, feat2_list)
        return feat1_cap, feat2_cap, seg_pre

    @torch.no_grad()
    def generate_change_caption(self, path_A, path_B):
        if not self.caption_enabled:
            print('Caption disabled (missing matching vocab for this checkpoint).')
            return ''
        imgA, imgB = self._read_and_preprocess(path_A, path_B)
        f1, f2, _ = self._forward_once(imgA, imgB)
        seq = self.decoder.sample_beam(f1, f2, k=self.args.beam_size) if self.args.beam_size > 1 else self.decoder.sample(f1, f2, k=1)
        specials = {self.word_vocab['<START>'], self.word_vocab['<END>'], self.word_vocab['<NULL>']}
        if isinstance(seq, list) and (len(seq) > 0) and isinstance(seq[0], list):
            seq = seq[0]
        tokens = [self.idx_to_word.get(i, '<UNK>') for i in seq if i not in specials]
        caption = ' '.join(tokens).strip()
        print('Caption:', caption)
        return caption

    @torch.no_grad()
    def change_detection(self, path_A, path_B, savepath_mask):
        imgA, imgB = self._read_and_preprocess(path_A, path_B)
        _, _, seg_pre = self._forward_once(imgA, imgB)
        pred = np.argmax(seg_pre.detach().cpu().numpy(), axis=1)[0].astype(np.uint8)

        # Colorize in RGB: road=yellow (255,255,0), building=red (255,0,0)
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = (255, 255, 0)
        pred_rgb[pred == 2] = (255, 0, 0)
        os.makedirs(os.path.dirname(savepath_mask) or '.', exist_ok=True)
        cv2.imwrite(savepath_mask, pred_rgb[:, :, ::-1])  # convert RGB->BGR for OpenCV
        print('Mask saved:', savepath_mask)
        return pred

if __name__ == '__main__':
    cp = Change_Perception()
    _ = cp.generate_change_caption(cp.args.imgA_path, cp.args.imgB_path)
    _ = cp.change_detection(cp.args.imgA_path, cp.args.imgB_path, cp.args.mask_save_path)