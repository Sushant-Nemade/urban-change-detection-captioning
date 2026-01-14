# # 
# ############## To resolve the above error the below code is with args short save names and default fallback soln for long file path error #####
"""

    Exact code changes (full updated train_newoptimized.py) adding:
        cap_unfreeze_encoder_trans flag: optionally unfreeze only the caption modules in encoder_trans during goal=1 (default keeps it frozen as now).
        det_freeze_caption_in_encoder_trans flag: optionally isolate detection modules during goal=0 (default keeps current behavior).
        use_loss_balance flag: optionally enable loss balancing for joint training (goal=2) (default off).
        set_modes adjusted so we don’t override the fine-grained train/eval states when the above flags are used.
    Commands for Step A (without loss balance) and Step B (with loss balance), using your epoch-17 checkpoint.

What the updated solution does

    When goal=1 and --cap_unfreeze_encoder_trans 1:
        Only the captioning-related submodules inside AttentiveEncoder are unfrozen and set to train; detection submodules are frozen and set to eval. This adapts the change-fusion to your augmented captions without touching detection.
        Decoder stays trainable; encoder stays frozen (as before).
        Validation can use beam_size=3 for more realistic BLEU-4.
    When goal=0:
        By default, behavior is unchanged (as you requested). Optionally, if you pass --det_freeze_caption_in_encoder_trans 1, only detection submodules are trainable and captioning submodules are frozen/eval; decoder is frozen.
    When goal=2 and --use_loss_balance 1:
        det_loss and cap_loss are normalized by their detached values before summing, following the paper’s balancing strategy. This stabilizes joint training on your augmented set.
        Recommended to run 5–10 epochs max with early stopping patience 5–10.

"""

# # train_newoptimized.py
# import argparse
# import json
# import os
# import time
# import random
# import numpy as np
# import torch
# import torch.optim
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.utils import data
# from torch.cuda.amp import autocast, GradScaler
# from tqdm import tqdm
# 
# from data.LEVIR_MCI import LEVIRCCDataset
# from model.model_encoder_att import Encoder, AttentiveEncoder
# from model.model_decoder import DecoderTransformer
# from utils_tool.utils import *
# from utils_tool.metrics import Evaluator
# 
# # Speed opts for RTX 4090
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# try:
#     torch.set_float32_matmul_precision('high')
# except Exception:
#     pass
# 
# 
# def metric_for_goal(train_goal, bleu4, miou):
#     if train_goal == 1:
#         return bleu4
#     if train_goal == 0:
#         return miou
#     return bleu4 + miou
# 
# 
# def transplant_decoder_vocab_rows(decoder_module, checkpoint_decoder_sd, source_vocab_path, target_vocab_dict, log_fn=print):
#     """
#     Copy overlapping tokens' embedding and classifier rows from the source checkpoint
#     (original vocab) into the current decoder (augmented vocab) to speed convergence.
#     Safe: only writes to decoder_module; does not modify any checkpoint files.
#     """
#     if not os.path.exists(source_vocab_path):
#         log_fn("Vocab transplant: source_vocab_json not found; skipping.")
#         return 0, 0
# 
#     with open(source_vocab_path, 'r') as f:
#         src_tok2idx = json.load(f)  # token -> int
#     tgt_tok2idx = target_vocab_dict  # token -> int
# 
#     common = set(src_tok2idx.keys()) & set(tgt_tok2idx.keys())
#     if len(common) == 0:
#         log_fn("Vocab transplant: no overlapping tokens; skipping.")
#         return 0, 0
# 
#     old_emb = checkpoint_decoder_sd.get('vocab_embedding.weight', None)
#     old_w = checkpoint_decoder_sd.get('wdc.weight', None)
#     old_b = checkpoint_decoder_sd.get('wdc.bias', None)
#     if old_emb is None or old_w is None or old_b is None:
#         log_fn("Vocab transplant: missing embedding or classifier in checkpoint; skipping.")
#         return 0, 0
# 
#     new_emb = decoder_module.vocab_embedding.weight.data
#     new_w = decoder_module.wdc.weight.data
#     new_b = decoder_module.wdc.bias.data
# 
#     copied = 0
#     for tok in common:
#         si = src_tok2idx[tok]
#         ti = tgt_tok2idx[tok]
#         if si < old_emb.shape[0] and ti < new_emb.shape[0]:
#             new_emb[ti].copy_(old_emb[si]); copied += 1
#         if si < old_w.shape[0] and ti < new_w.shape[0]:
#             new_w[ti].copy_(old_w[si])
#         if si < old_b.shape[0] and ti < new_b.shape[0]:
#             new_b[ti].copy_(old_b[si])
# 
#     log_fn(f"Vocab transplant: copied {copied} overlapping tokens from source -> target.")
#     return copied, len(common)
# 
# 
# def safe_torch_save(state, dir_path, preferred_filename, fallback_filename, log_fn=print):
#     """
#     Safe save for Windows:
#       - Try the preferred (long) filename.
#       - On failure (e.g., MAX_PATH), retry with a compact fallback filename.
#     Returns the path actually written.
#     """
#     os.makedirs(dir_path, exist_ok=True)
#     full_pref = os.path.join(dir_path, preferred_filename)
#     try:
#         torch.save(state, full_pref)
#         return full_pref
#     except Exception as e:
#         log_fn(f"Warning: save failed for '{full_pref}'. Retrying with compact name. Error: {e}")
#         full_fallback = os.path.join(dir_path, fallback_filename)
#         torch.save(state, full_fallback)
#         return full_fallback
# 
# 
# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         self.current_goal = args.train_goal
# 
#         run_name = f'baseline_{time_file_str()}_train_goal_{args.train_goal}_{random.randint(10, 100)}'
#         self.args.savepath = os.path.join(args.savepath, run_name)
#         os.makedirs(self.args.savepath, exist_ok=True)
#         self.log = open(os.path.join(self.args.savepath, f'{run_name}.log'), 'w')
# 
#         print_log(f'=>datset: {args.data_name}', self.log)
#         print_log(f'=>network: {args.network}', self.log)
#         print_log(f'=>encoder_lr: {args.encoder_lr}', self.log)
#         print_log(f'=>decoder_lr: {args.decoder_lr}', self.log)
#         print_log(f'=>num_epochs: {args.num_epochs}', self.log)
#         print_log(f'=>train_batchsize: {args.train_batchsize}', self.log)
# 
#         with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
#             self.word_vocab = json.load(f)
# 
#         self.build_model()
# 
#         self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
#         self.criterion_det = torch.nn.CrossEntropyLoss().cuda()
#         self.scaler = GradScaler(enabled=torch.cuda.is_available())
# 
#         # DataLoaders
#         if args.data_name == 'LEVIR_MCI':
#             self.train_loader = data.DataLoader(
#                 LEVIRCCDataset(args.data_folder, args.list_path, 'train',
#                                args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
#                 batch_size=args.train_batchsize, shuffle=True,
#                 num_workers=args.num_workers, pin_memory=True, drop_last=True,
#                 persistent_workers=(args.num_workers > 0)
#             )
#             self.val_loader = data.DataLoader(
#                 LEVIRCCDataset(args.data_folder, args.list_path, 'val',
#                                args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
#                 batch_size=args.val_batchsize, shuffle=False,
#                 num_workers=0, pin_memory=True
#             )
# 
#         self.index_i = 0
#         self.hist = np.zeros((max(1, args.num_epochs) * max(1, len(self.train_loader)), 5))
#         self.best_bleu4 = 0.0
#         self.MIou = 0.0
#         self.Sum_Metric = 0.0
#         self.best_epoch = -1
#         self.best_model_path = None
#         self.no_improve_epochs = 0
# 
#         self.evaluator = Evaluator(num_class=3)
# 
#     def build_model(self):
#         args = self.args
#         self.encoder = Encoder(args.network)
#         self.encoder_trans = AttentiveEncoder(
#             args.train_stage, args.n_layers,
#             (args.feat_size, args.feat_size, args.encoder_dim),
#             args.n_heads
#         )
#         self.decoder = DecoderTransformer(
#             encoder_dim=args.encoder_dim,
#             feature_dim=args.feature_dim,
#             vocab_size=len(self.word_vocab),
#             max_lengths=args.max_length,
#             word_vocab=self.word_vocab,
#             n_head=args.n_heads,
#             n_layers=args.decoder_n_layers,
#             dropout=args.dropout
#         )
# 
#         fine_tune_cap_decoder = True
#         if args.train_stage == 's1':
#             args.fine_tune_encoder = True
#             self.encoder.fine_tune(args.fine_tune_encoder)
#             self.encoder_trans.fine_tune(self.current_goal)
#             self.decoder.fine_tune(fine_tune_cap_decoder)
# 
#         elif args.train_stage == 's2':
#             if args.checkpoint is None or not os.path.exists(args.checkpoint):
#                 raise ValueError('Error: checkpoint must be set for train_stage=s2.')
# 
#             checkpoint = torch.load(args.checkpoint, map_location='cuda')
#             print_log(f'Load Model from {args.checkpoint}', self.log)
# 
#             self.encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
#             self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
# 
#             dec_sd_ckpt = checkpoint['decoder_dict']
#             model_sd = self.decoder.state_dict()
#             skip = 0
#             # Load transformer layers etc. (shape-matching only)
#             for k, v in dec_sd_ckpt.items():
#                 if k in model_sd and model_sd[k].shape == v.shape and not k.startswith('vocab_embedding') and not k.startswith('wdc.'):
#                     model_sd[k].copy_(v)
#                 elif k in model_sd and model_sd[k].shape == v.shape:
#                     # If shapes match for vocab layers (same vocab), copy too
#                     model_sd[k].copy_(v)
#                 else:
#                     skip += 1
#             self.decoder.load_state_dict(model_sd, strict=False)
#             if skip > 0:
#                 print_log(f'Note: skipped {skip} decoder params due to shape mismatch (likely vocab change).', self.log)
# 
#             # If vocab size differs and source vocab provided, transplant overlapping tokens
#             old_vocab_size = dec_sd_ckpt.get('wdc.weight', torch.empty(0)).shape[0] if 'wdc.weight' in dec_sd_ckpt else None
#             new_vocab_size = self.decoder.wdc.weight.shape[0]
#             if args.source_vocab_json and os.path.exists(args.source_vocab_json) and old_vocab_size and new_vocab_size != old_vocab_size:
#                 transplant_decoder_vocab_rows(
#                     decoder_module=self.decoder,
#                     checkpoint_decoder_sd=dec_sd_ckpt,
#                     source_vocab_path=args.source_vocab_json,
#                     target_vocab_dict=self.word_vocab,
#                     log_fn=lambda m: print_log(m, self.log)
#                 )
#             else:
#                 print_log("Vocab transplant: skipped (no source_vocab_json provided or vocab sizes match).", self.log)
# 
#             # Fine-tune flags
#             if args.train_goal == 2:
#                 args.fine_tune_encoder = True
#                 self.encoder.fine_tune(True)
#                 for p in self.encoder_trans.parameters():
#                     p.requires_grad = True
#                 fine_tune_cap_decoder = True
#             elif args.train_goal == 1:
#                 args.fine_tune_encoder = False
#                 self.encoder.fine_tune(False)
#                 for p in self.encoder_trans.parameters():
#                     p.requires_grad = False
#                 fine_tune_cap_decoder = True
#             else:
#                 args.fine_tune_encoder = True
#                 self.encoder.fine_tune(True)
#                 for p in self.encoder_trans.parameters():
#                     p.requires_grad = True
#                 fine_tune_cap_decoder = False
#             self.decoder.fine_tune(fine_tune_cap_decoder)
#         else:
#             raise ValueError('train_stage must be s1 or s2.')
# 
#         self.encoder = self.encoder.cuda()
#         self.encoder_trans = self.encoder_trans.cuda()
#         self.decoder = self.decoder.cuda()
# 
#         if getattr(args, 'use_compile', False):
#             self.encoder_trans = torch.compile(self.encoder_trans)
#             self.decoder = torch.compile(self.decoder)
# 
#         self.encoder_optimizer = torch.optim.Adam(
#             params=self.encoder.parameters(), lr=args.encoder_lr
#         ) if args.fine_tune_encoder else None
# 
#         self.encoder_trans_optimizer = torch.optim.Adam(
#             params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
#             lr=args.encoder_lr
#         )
# 
#         self.decoder_optimizer = torch.optim.Adam(
#             params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
#             lr=args.decoder_lr
#         ) if fine_tune_cap_decoder else None
# 
#         from torch.optim.lr_scheduler import ReduceLROnPlateau
#         factor = args.lr_factor
#         patience = args.lr_patience
#         self.encoder_lr_scheduler = ReduceLROnPlateau(
#             self.encoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True
#         ) if self.encoder_optimizer is not None else None
#         self.encoder_trans_lr_scheduler = ReduceLROnPlateau(
#             self.encoder_trans_optimizer, mode='max', factor=factor, patience=patience, verbose=True
#         )
#         self.decoder_lr_scheduler = ReduceLROnPlateau(
#             self.decoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True
#         ) if self.decoder_optimizer is not None else None
# 
#     def set_modes(self):
#         g = self.current_goal
#         if g == 2:
#             self.encoder.train(); self.encoder_trans.train(); self.decoder.train()
#         elif g == 1:
#             self.encoder.eval(); self.encoder_trans.eval(); self.decoder.train()
#         else:
#             self.encoder.train(); self.encoder_trans.train(); self.decoder.eval()
# 
#     def zero_all(self):
#         if self.decoder_optimizer is not None:
#             self.decoder_optimizer.zero_grad(set_to_none=True)
#         self.encoder_trans_optimizer.zero_grad(set_to_none=True)
#         if self.encoder_optimizer is not None:
#             self.encoder_optimizer.zero_grad(set_to_none=True)
# 
#     def unscale_and_clip(self):
#         max_norm = self.args.grad_clip
#         if max_norm is None:
#             return
#         if self.decoder_optimizer is not None:
#             self.scaler.unscale_(self.decoder_optimizer)
#         self.scaler.unscale_(self.encoder_trans_optimizer)
#         if self.encoder_optimizer is not None:
#             self.scaler.unscale_(self.encoder_optimizer)
#         if self.decoder_optimizer is not None:
#             torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm)
#         torch.nn.utils.clip_grad_norm_(self.encoder_trans.parameters(), max_norm)
#         if self.encoder_optimizer is not None:
#             torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm)
# 
#     def training(self, epoch):
#         args = self.args
#         self.set_modes()
#         self.zero_all()
# 
#         running_det = 0.0
#         running_cap = 0.0
#         running_joint = 0.0
#         num_batches = 0
# 
#         accum_steps = max(1, 64 // max(1, args.train_batchsize))
# 
#         for i, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(self.train_loader):
#             t0 = time.time()
# 
#             imgA = imgA.cuda(non_blocking=True)
#             imgB = imgB.cuda(non_blocking=True)
#             seg_label = seg_label.cuda(non_blocking=True)
#             token = token.squeeze(1).cuda(non_blocking=True)
#             token_len = token_len.cuda(non_blocking=True)
# 
#             with autocast(enabled=torch.cuda.is_available()):
#                 feat1, feat2 = self.encoder(imgA, imgB)
#                 feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
# 
#                 det_loss = self.criterion_det(seg_pre, seg_label.to(torch.int64))
# 
#                 cap_loss = None
#                 if self.current_goal != 0:
#                     scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len)
#                     targets = caps_sorted[:, 1:]
#                     scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
#                     targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
#                     cap_loss = self.criterion_cap(scores, targets.to(torch.int64))
# 
#                 if self.current_goal == 0:
#                     loss = det_loss
#                 elif self.current_goal == 1:
#                     loss = cap_loss
#                 else:
#                     loss = det_loss + (cap_loss if cap_loss is not None else 0.0)
# 
#                 loss = loss / accum_steps
# 
#             self.scaler.scale(loss).backward()
# 
#             # Step only at accumulation boundary
#             boundary = ((i + 1) % accum_steps == 0) or ((i + 1) == len(self.train_loader))
#             if boundary:
#                 self.unscale_and_clip()
#                 if self.decoder_optimizer is not None:
#                     self.scaler.step(self.decoder_optimizer)
#                 self.scaler.step(self.encoder_trans_optimizer)
#                 if self.encoder_optimizer is not None:
#                     self.scaler.step(self.encoder_optimizer)
#                 self.scaler.update()
#                 self.zero_all()
# 
#             running_det += float(det_loss.item())
#             if cap_loss is not None:
#                 running_cap += float(cap_loss.item())
#             running_joint += float(loss.item()) * accum_steps
#             num_batches += 1
# 
#             self.hist[self.index_i, 0] = time.time() - t0
#             self.hist[self.index_i, 1] = float(det_loss.item())
#             self.hist[self.index_i, 2] = accuracy(
#                 seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
#                 seg_label.reshape(-1), 1
#             )
#             if cap_loss is not None:
#                 self.hist[self.index_i, 3] = float(cap_loss.item())
#                 self.hist[self.index_i, 4] = accuracy(scores, targets, 5)
#             self.index_i += 1
# 
#             if self.index_i % args.print_freq == 0:
#                 w0, w1 = max(0, self.index_i - args.print_freq), self.index_i
#                 print_log(
#                     'Training Epoch: [{0}][{1}/{2}]  '
#                     'Avg Batch Time: {3:.3f}  Det_Loss: {4:.4f}  Det Acc: {5:.3f}  '
#                     'Cap_loss: {6:.5f}  Text_Top-5 Acc: {7:.3f}'
#                     .format(
#                         epoch, i, len(self.train_loader),
#                         np.mean(self.hist[w0:w1, 0]),
#                         np.mean(self.hist[w0:w1, 1]),
#                         np.mean(self.hist[w0:w1, 2]),
#                         np.mean(self.hist[w0:w1, 3]),
#                         np.mean(self.hist[w0:w1, 4])
#                     ), self.log
#                 )
# 
#         avg_det = running_det / max(1, num_batches)
#         avg_cap = running_cap / max(1, num_batches)
#         avg_joint = running_joint / max(1, num_batches)
#         print_log(f'Epoch {epoch} Avg - Det Loss: {avg_det:.4f}, Cap Loss: {avg_cap:.4f}, Joint Loss: {avg_joint:.4f}', self.log)
# 
#     def validation(self, epoch):
#         word_vocab = self.word_vocab
#         self.decoder.eval(); self.encoder_trans.eval(); self.encoder.eval()
# 
#         val_start_time = time.time()
#         references, hypotheses = [], []
#         self.evaluator.reset()
# 
#         with torch.no_grad():
#             for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _) in enumerate(
#                     tqdm(self.val_loader, desc=f'val_BEAM {self.args.beam_size}')):
#                 imgA = imgA.cuda(non_blocking=True)
#                 imgB = imgB.cuda(non_blocking=True)
#                 token_all = token_all.squeeze(0).cuda(non_blocking=True)
# 
#                 feat1, feat2 = self.encoder(imgA, imgB)
#                 feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)
# 
#                 if self.current_goal != 0:
#                     if self.args.beam_size > 1:
#                         seq = self.decoder.sample_beam(feat1, feat2, k=self.args.beam_size)
#                     else:
#                         seq = self.decoder.sample(feat1, feat2, k=1)
# 
#                 if self.current_goal != 1:
#                     pred_seg = seg_pre.data.cpu().numpy()
#                     seg_label_np = seg_label.cpu().numpy()
#                     pred_seg = np.argmax(pred_seg, axis=1)
#                     self.evaluator.add_batch(seg_label_np, pred_seg)
# 
#                 if self.current_goal != 0:
#                     img_token = token_all.tolist()
#                     img_tokens = list(map(lambda c: [w for w in c if w not in {
#                         word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
#                     references.append(img_tokens)
#                     pred_seq = [w for w in seq if w not in {
#                         word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
#                     hypotheses.append(pred_seq)
#                     assert len(references) == len(hypotheses)
# 
#         val_time = time.time() - val_start_time
# 
#         Acc_seg = Acc_class_seg = mIoU_seg = FWIoU_seg = 0.0
#         if self.current_goal != 1:
#             Acc_seg = self.evaluator.Pixel_Accuracy()
#             Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
#             mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
#             FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
#             print_log(
#                 '\nDetection_Validation:\n'
#                 f'Acc_seg: {Acc_seg:.5f}  Acc_class_seg: {Acc_class_seg:.5f}  '
#                 f'mIoU_seg: {mIoU_seg:.5f}  FWIoU_seg: {FWIoU_seg:.5f}', self.log
#             )
#             print_log(f'Iou: {IoU}', self.log)
# 
#         Bleu_1 = Bleu_2 = Bleu_3 = Bleu_4 = Meteor = Rouge = Cider = 0.0
#         if self.current_goal != 0:
#             score_dict = get_eval_score(references, hypotheses)
#             Bleu_1 = score_dict['Bleu_1']; Bleu_2 = score_dict['Bleu_2']
#             Bleu_3 = score_dict['Bleu_3']; Bleu_4 = score_dict['Bleu_4']
#             Meteor = score_dict['METEOR']; Rouge = score_dict['ROUGE_L']; Cider = score_dict['CIDEr']
#             print_log('Captioning_Validation:\n'
#                       f'Time: {val_time:.3f}  BLEU-1: {Bleu_1:.5f}  BLEU-2: {Bleu_2:.5f}  BLEU-3: {Bleu_3:.5f}  '
#                       f'BLEU-4: {Bleu_4:.5f}  Meteor: {Meteor:.5f}  Rouge: {Rouge:.5f}  Cider: {Cider:.5f}', self.log)
# 
#         metric_value = metric_for_goal(self.current_goal, Bleu_4, mIoU_seg)
#         if self.encoder_lr_scheduler is not None:
#             self.encoder_lr_scheduler.step(metric_value)
#         self.encoder_trans_lr_scheduler.step(metric_value)
#         if self.decoder_lr_scheduler is not None:
#             self.decoder_lr_scheduler.step(metric_value)
# 
#         improved = False
#         if self.current_goal == 2:
#             improved = (Bleu_4 + mIoU_seg) >= self.Sum_Metric
#             if improved:
#                 self.Sum_Metric = Bleu_4 + mIoU_seg
#         elif self.current_goal == 1:
#             improved = Bleu_4 >= self.best_bleu4
#             if improved:
#                 self.best_bleu4 = Bleu_4
#         else:
#             improved = mIoU_seg >= self.MIou
#             if improved:
#                 self.MIou = mIoU_seg
# 
#         if improved:
#             self.no_improve_epochs = 0
#             print('Save Model')
#             state = {
#                 'encoder_dict': self.encoder.state_dict(),
#                 'encoder_trans_dict': self.encoder_trans.state_dict(),
#                 'decoder_dict': self.decoder.state_dict()
#             }
# 
#             # Preferred (long) name or compact name based on short_save_names
#             metric = f'Sum_{round(100000 * (Bleu_4 + mIoU_seg))}_MIou_{round(100000 * mIoU_seg)}_Bleu4_{round(100000 * Bleu_4)}'
#             if self.args.short_save_names:
#                 preferred_name = f'{self.args.data_name}_epo_{epoch}_sum{(Bleu_4 + mIoU_seg):.3f}_miou{mIoU_seg:.3f}_bleu{Bleu_4:.3f}.pth'
#             else:
#                 preferred_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
# 
#             # Fallback compact name if preferred fails due to Windows path length
#             fallback_name = f'{self.args.data_name}_epo_{epoch}.pth'
# 
#             best_model_path = safe_torch_save(
#                 state, self.args.savepath, preferred_name, fallback_name,
#                 log_fn=lambda m: print_log(m, self.log)
#             )
#             self.best_epoch = epoch
#             self.best_model_path = best_model_path
#         else:
#             self.no_improve_epochs += 1
# 
#         enc_lr = self.encoder_optimizer.param_groups[0]['lr'] if self.encoder_optimizer is not None else 0.0
#         enc_tr_lr = self.encoder_trans_optimizer.param_groups[0]['lr']
#         dec_lr = self.decoder_optimizer.param_groups[0]['lr'] if self.decoder_optimizer is not None else 0.0
#         print_log(f'LRs — encoder: {enc_lr:.6f}, encoder_trans: {enc_tr_lr:.6f}, decoder: {dec_lr:.6f}', self.log)
# 
#     def fit(self):
#         for epoch in range(self.args.start_epoch, self.args.num_epochs):
#             self.training(epoch)
#             if (epoch % self.args.val_interval == 0) or (epoch == self.args.num_epochs - 1):
#                 self.validation(epoch)
#             if self.no_improve_epochs >= self.args.early_stop_patience:
#                 print_log(f'Early stopping at epoch {epoch} (no improvement for {self.args.early_stop_patience} epochs)', self.log)
#                 break
# 
# 
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
# 
#     # Data
#     parser.add_argument('--data_folder', default=r'C:\Users\nemade\Downloads\LEVIR-MCI-data\LEVIR-MCI-dataset\images')
#     parser.add_argument('--list_path', default='./data/LEVIR_MCI/')
#     parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/')
#     parser.add_argument('--vocab_file', default='vocab')  # augmented vocab by default
#     parser.add_argument('--source_vocab_json', type=str, default='', help='path to original vocab (token->idx) for transplant')
#     parser.add_argument('--max_length', type=int, default=46)  # set to 46 per your preprocess
#     parser.add_argument('--allow_unk', type=int, default=1)
#     parser.add_argument('--data_name', default='LEVIR_MCI')
# 
#     # Runtime
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--print_freq', type=int, default=100)
#     parser.add_argument('--val_interval', type=int, default=1)
#     parser.add_argument('--start_epoch', type=int, default=0)
#     parser.add_argument('--savepath', default='./models_ckpt/')
#     parser.add_argument('--beam_size', type=int, default=1, help='validation beam size (batch must be 1)')
#     parser.add_argument('--short_save_names', type=bool, default=False, help='use compact checkpoint filenames')
# 
#     # Stage/goal/checkpoint
#     parser.add_argument('--train_goal', type=int, default=2, help='0: det; 1: cap; 2: joint')
#     parser.add_argument('--train_stage', default='s2', help='s1: pretrain; s2: finetune from checkpoint')
#     parser.add_argument('--checkpoint', required=True, help='path to checkpoint used for s2')
# 
#     # Performance
#     parser.add_argument('--num_workers', type=int, default=0)
#     parser.add_argument('--use_compile', type=bool, default=False)
# 
#     # LRs and clipping
#     parser.add_argument('--encoder_lr', type=float, default=1e-4)
#     parser.add_argument('--decoder_lr', type=float, default=2e-4)
#     parser.add_argument('--grad_clip', type=float, default=1.0)
# 
#     # Schedulers
#     parser.add_argument('--lr_factor', type=float, default=0.5)
#     parser.add_argument('--lr_patience', type=int, default=5)
#     parser.add_argument('--early_stop_patience', type=int, default=20)
# 
#     # Model params
#     parser.add_argument('--train_batchsize', type=int, default=32)
#     parser.add_argument('--val_batchsize', type=int, default=1)
#     parser.add_argument('--num_epochs', type=int, default=50)
#     parser.add_argument('--dropout', type=float, default=0.1)
#     parser.add_argument('--network', default='segformer-mit_b1')
#     parser.add_argument('--encoder_dim', type=int, default=512)
#     parser.add_argument('--feat_size', type=int, default=16)
#     parser.add_argument('--n_heads', type=int, default=8)
#     parser.add_argument('--n_layers', type=int, default=3)
#     parser.add_argument('--decoder_n_layers', type=int, default=1)
#     parser.add_argument('--feature_dim', type=int, default=512)
# 
#     args = parser.parse_args()
#     torch.cuda.set_device(args.gpu_id)
#     trainer = Trainer(args)
#     print('Starting Epoch:', args.start_epoch)
#     print('Total Epochs:', args.num_epochs)
#     trainer.fit()
# 
# 
# 
# ####################### Below code is specifically for optimization of  the  above code which normally trains the whole model ##############
#
# # train_newoptimized.py
# import argparse
# import json
# import os
# import time
# import random
# import numpy as np
# import torch
# import torch.optim
# from torch.nn.utils.rnn import pack_padded_sequence
# from torch.utils import data
# from torch.cuda.amp import autocast, GradScaler
# from tqdm import tqdm
#
# from data.LEVIR_MCI import LEVIRCCDataset
# from model.model_encoder_att import Encoder, AttentiveEncoder
# from model.model_decoder import DecoderTransformer
# from utils_tool.utils import *
# from utils_tool.metrics import Evaluator
#
# # Speed opts for modern GPUs
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# try:
#     torch.set_float32_matmul_precision('high')
# except Exception:
#     pass
#
#
# def metric_for_goal(train_goal, bleu4, miou):
#     if train_goal == 1:
#         return bleu4
#     if train_goal == 0:
#         return miou
#     return bleu4 + miou
#
#
# def transplant_decoder_vocab_rows(decoder_module, checkpoint_decoder_sd, source_vocab_path, target_vocab_dict, log_fn=print):
#     """
#     Copy overlapping tokens (embedding and classifier rows) from source checkpoint decoder
#     into current decoder vocabulary to speed convergence on augmented vocab.
#     """
#     if not source_vocab_path or not os.path.exists(source_vocab_path):
#         log_fn("Vocab transplant: source_vocab_json not found; skipping.")
#         return 0, 0
#
#     with open(source_vocab_path, 'r') as f:
#         src_tok2idx = json.load(f)  # token -> idx
#     tgt_tok2idx = target_vocab_dict  # token -> idx
#
#     common = set(src_tok2idx.keys()) & set(tgt_tok2idx.keys())
#     if len(common) == 0:
#         log_fn("Vocab transplant: no overlapping tokens; skipping.")
#         return 0, 0
#
#     old_emb = checkpoint_decoder_sd.get('vocab_embedding.weight', None)
#     old_w = checkpoint_decoder_sd.get('wdc.weight', None)
#     old_b = checkpoint_decoder_sd.get('wdc.bias', None)
#     if old_emb is None or old_w is None or old_b is None:
#         log_fn("Vocab transplant: missing embedding or classifier in checkpoint; skipping.")
#         return 0, 0
#
#     new_emb = decoder_module.vocab_embedding.weight.data
#     new_w = decoder_module.wdc.weight.data
#     new_b = decoder_module.wdc.bias.data
#
#     copied = 0
#     for tok in common:
#         si = src_tok2idx[tok]
#         ti = tgt_tok2idx[tok]
#         if si < old_emb.shape[0] and ti < new_emb.shape[0]:
#             new_emb[ti].copy_(old_emb[si]); copied += 1
#         if si < old_w.shape[0] and ti < new_w.shape[0]:
#             new_w[ti].copy_(old_w[si])
#         if si < old_b.shape[0] and ti < new_b.shape[0]:
#             new_b[ti].copy_(old_b[si])
#
#     log_fn(f"Vocab transplant: copied {copied} overlapping tokens from source -> target.")
#     return copied, len(common)
#
#
# def safe_torch_save(state, dir_path, preferred_filename, fallback_filename, log_fn=print):
#     """
#     Safe save for Windows long paths:
#     - Try preferred filename; if it fails, use a compact fallback.
#     """
#     os.makedirs(dir_path, exist_ok=True)
#     full_pref = os.path.join(dir_path, preferred_filename)
#     try:
#         torch.save(state, full_pref)
#         return full_pref
#     except Exception as e:
#         log_fn(f"Warning: save failed for '{full_pref}'. Retrying with compact name. Error: {e}")
#         full_fallback = os.path.join(dir_path, fallback_filename)
#         torch.save(state, full_fallback)
#         return full_fallback
#
#
# class Trainer(object):
#     def __init__(self, args):
#         self.args = args
#         self.current_goal = args.train_goal
#
#         run_name = f'baseline_{time_file_str()}_train_goal_{args.train_goal}_{random.randint(10, 100)}'
#         self.args.savepath = os.path.join(args.savepath, run_name)
#         os.makedirs(self.args.savepath, exist_ok=True)
#         self.log = open(os.path.join(self.args.savepath, f'{run_name}.log'), 'w')
#
#         print_log(f'=>datset: {args.data_name}', self.log)
#         print_log(f'=>network: {args.network}', self.log)
#         print_log(f'=>encoder_lr: {args.encoder_lr}', self.log)
#         print_log(f'=>decoder_lr: {args.decoder_lr}', self.log)
#         print_log(f'=>num_epochs: {args.num_epochs}', self.log)
#         print_log(f'=>train_batchsize: {args.train_batchsize}', self.log)
#         if args.use_loss_balance and args.train_goal == 2:
#             print_log('=>loss_balance: ON (normalize det and cap losses)', self.log)
#         else:
#             print_log('=>loss_balance: OFF', self.log)
#
#         # Load vocab
#         with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
#             self.word_vocab = json.load(f)
#
#         self.build_model()
#
#         self.criterion_cap = torch.nn.CrossEntropyLoss().cuda()
#         self.criterion_det = torch.nn.CrossEntropyLoss().cuda()
#         self.scaler = GradScaler(enabled=torch.cuda.is_available())
#
#         # DataLoaders
#         if args.data_name == 'LEVIR_MCI':
#             self.train_loader = data.DataLoader(
#                 LEVIRCCDataset(args.data_folder, args.list_path, 'train',
#                                args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
#                 batch_size=args.train_batchsize, shuffle=True,
#                 num_workers=args.num_workers, pin_memory=True, drop_last=True,
#                 persistent_workers=(args.num_workers > 0)
#             )
#             self.val_loader = data.DataLoader(
#                 LEVIRCCDataset(args.data_folder, args.list_path, 'val',
#                                args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
#                 batch_size=args.val_batchsize, shuffle=False,
#                 num_workers=0, pin_memory=True
#             )
#
#         # Stats
#         self.index_i = 0
#         self.hist = np.zeros((max(1, args.num_epochs) * max(1, len(self.train_loader)), 5))
#         self.best_bleu4 = 0.0
#         self.MIou = 0.0
#         self.Sum_Metric = 0.0
#         self.best_epoch = -1
#         self.best_model_path = None
#         self.no_improve_epochs = 0
#
#         self.evaluator = Evaluator(num_class=3)
#
#     def build_model(self):
#         args = self.args
#         self.encoder = Encoder(args.network)
#         self.encoder_trans = AttentiveEncoder(
#             args.train_stage, args.n_layers,
#             (args.feat_size, args.feat_size, args.encoder_dim),
#             args.n_heads
#         )
#         self.decoder = DecoderTransformer(
#             encoder_dim=args.encoder_dim,
#             feature_dim=args.feature_dim,
#             vocab_size=len(self.word_vocab),
#             max_lengths=args.max_length,
#             word_vocab=self.word_vocab,
#             n_head=args.n_heads,
#             n_layers=args.decoder_n_layers,
#             dropout=args.dropout
#         )
#
#         # Whether we let encoder_trans manage its own train/eval state (partial unfreeze)
#         self.manage_encoder_trans_mode = False
#
#         fine_tune_cap_decoder = True
#         if args.train_stage == 's1':
#             args.fine_tune_encoder = True
#             self.encoder.fine_tune(args.fine_tune_encoder)
#             # s1: follow goal to fine-tune encoder_trans
#             self.encoder_trans.fine_tune(self.current_goal)
#             self.manage_encoder_trans_mode = True
#             self.decoder.fine_tune(fine_tune_cap_decoder)
#
#         elif args.train_stage == 's2':
#             if args.checkpoint is None or not os.path.exists(args.checkpoint):
#                 raise ValueError('Error: checkpoint must be set for train_stage=s2.')
#
#             checkpoint = torch.load(args.checkpoint, map_location='cuda')
#             print_log(f'Load Model from {args.checkpoint}', self.log)
#
#             self.encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
#             self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)
#
#             dec_sd_ckpt = checkpoint['decoder_dict']
#             model_sd = self.decoder.state_dict()
#             skip = 0
#             for k, v in dec_sd_ckpt.items():
#                 if k in model_sd and model_sd[k].shape == v.shape and not k.startswith('vocab_embedding') and not k.startswith('wdc.'):
#                     model_sd[k].copy_(v)
#                 elif k in model_sd and model_sd[k].shape == v.shape:
#                     model_sd[k].copy_(v)
#                 else:
#                     skip += 1
#             self.decoder.load_state_dict(model_sd, strict=False)
#             if skip > 0:
#                 print_log(f'Note: skipped {skip} decoder params due to shape mismatch (likely vocab change).', self.log)
#
#             # If vocab size differs and source vocab provided, transplant overlapping tokens
#             old_vocab_size = dec_sd_ckpt.get('wdc.weight', torch.empty(0)).shape[0] if 'wdc.weight' in dec_sd_ckpt else None
#             new_vocab_size = self.decoder.wdc.weight.shape[0]
#             if args.source_vocab_json and os.path.exists(args.source_vocab_json) and old_vocab_size and new_vocab_size != old_vocab_size:
#                 transplant_decoder_vocab_rows(
#                     decoder_module=self.decoder,
#                     checkpoint_decoder_sd=dec_sd_ckpt,
#                     source_vocab_path=args.source_vocab_json,
#                     target_vocab_dict=self.word_vocab,
#                     log_fn=lambda m: print_log(m, self.log)
#                 )
#             else:
#                 print_log("Vocab transplant: skipped (no source_vocab_json provided or vocab sizes match).", self.log)
#
#             # Fine-tune per goal
#             if args.train_goal == 2:
#                 args.fine_tune_encoder = True
#                 self.encoder.fine_tune(True)
#                 for p in self.encoder_trans.parameters():
#                     p.requires_grad = True
#                 fine_tune_cap_decoder = True
#
#             elif args.train_goal == 1:
#                 args.fine_tune_encoder = False
#                 self.encoder.fine_tune(False)
#                 if args.cap_unfreeze_encoder_trans == 1:
#                     # Partially unfreeze: caption modules train, detection modules frozen/eval
#                     self.encoder_trans.fine_tune(1)
#                     self.manage_encoder_trans_mode = True
#                 else:
#                     # Default: keep encoder_trans fully frozen
#                     for p in self.encoder_trans.parameters():
#                         p.requires_grad = False
#                 fine_tune_cap_decoder = True
#
#             else:  # goal == 0
#                 args.fine_tune_encoder = True
#                 self.encoder.fine_tune(True)
#                 # Default behavior: encoder_trans trainable
#                 for p in self.encoder_trans.parameters():
#                     p.requires_grad = True
#                 fine_tune_cap_decoder = False
#
#             self.decoder.fine_tune(fine_tune_cap_decoder)
#         else:
#             raise ValueError('train_stage must be s1 or s2.')
#
#         # CUDA + optional compile
#         self.encoder = self.encoder.cuda()
#         self.encoder_trans = self.encoder_trans.cuda()
#         self.decoder = self.decoder.cuda()
#         if getattr(args, 'use_compile', False):
#             self.encoder_trans = torch.compile(self.encoder_trans)
#             self.decoder = torch.compile(self.decoder)
#
#         # Optimizers
#         self.encoder_optimizer = torch.optim.Adam(
#             params=self.encoder.parameters(), lr=args.encoder_lr
#         ) if args.fine_tune_encoder else None
#
#         self.encoder_trans_optimizer = torch.optim.Adam(
#             params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
#             lr=args.encoder_lr
#         )
#
#         self.decoder_optimizer = torch.optim.Adam(
#             params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
#             lr=args.decoder_lr
#         ) if fine_tune_cap_decoder else None
#
#         # LR schedulers
#         from torch.optim.lr_scheduler import ReduceLROnPlateau
#         factor = args.lr_factor
#         patience = args.lr_patience
#         self.encoder_lr_scheduler = ReduceLROnPlateau(
#             self.encoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True
#         ) if self.encoder_optimizer is not None else None
#         self.encoder_trans_lr_scheduler = ReduceLROnPlateau(
#             self.encoder_trans_optimizer, mode='max', factor=factor, patience=patience, verbose=True
#         )
#         self.decoder_lr_scheduler = ReduceLROnPlateau(
#             self.decoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True
#         ) if self.decoder_optimizer is not None else None
#
#     def set_modes(self):
#         g = self.current_goal
#         if g == 2:
#             self.encoder.train()
#             # Do not override encoder_trans if partial-unfreeze is managed
#             if not self.manage_encoder_trans_mode:
#                 self.encoder_trans.train()
#             self.decoder.train()
#         elif g == 1:
#             self.encoder.eval()
#             if not self.manage_encoder_trans_mode:
#                 self.encoder_trans.eval()
#             self.decoder.train()
#         else:  # g == 0
#             self.encoder.train()
#             if not self.manage_encoder_trans_mode:
#                 self.encoder_trans.train()
#             self.decoder.eval()
#
#     def zero_all(self):
#         if self.decoder_optimizer is not None:
#             self.decoder_optimizer.zero_grad(set_to_none=True)
#         self.encoder_trans_optimizer.zero_grad(set_to_none=True)
#         if self.encoder_optimizer is not None:
#             self.encoder_optimizer.zero_grad(set_to_none=True)
#
#     def unscale_and_clip(self):
#         max_norm = self.args.grad_clip
#         if max_norm is None:
#             return
#         if self.decoder_optimizer is not None:
#             self.scaler.unscale_(self.decoder_optimizer)
#         self.scaler.unscale_(self.encoder_trans_optimizer)
#         if self.encoder_optimizer is not None:
#             self.scaler.unscale_(self.encoder_optimizer)
#         if self.decoder_optimizer is not None:
#             torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm)
#         torch.nn.utils.clip_grad_norm_(self.encoder_trans.parameters(), max_norm)
#         if self.encoder_optimizer is not None:
#             torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm)
#
#     def training(self, epoch):
#         args = self.args
#         self.set_modes()
#         self.zero_all()
#
#         running_det = 0.0
#         running_cap = 0.0
#         running_joint = 0.0
#         num_batches = 0
#
#         accum_steps = max(1, 64 // max(1, args.train_batchsize))
#
#         for i, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(self.train_loader):
#             t0 = time.time()
#
#             imgA = imgA.cuda(non_blocking=True)
#             imgB = imgB.cuda(non_blocking=True)
#             seg_label = seg_label.cuda(non_blocking=True)
#             token = token.squeeze(1).cuda(non_blocking=True)
#             token_len = token_len.cuda(non_blocking=True)
#
#             with autocast(enabled=torch.cuda.is_available()):
#                 feat1_list, feat2_list = self.encoder(imgA, imgB)
#                 feat1, feat2, seg_pre = self.encoder_trans(feat1_list, feat2_list)
#
#                 det_loss = self.criterion_det(seg_pre, seg_label.to(torch.int64))
#
#                 cap_loss = None
#                 if self.current_goal != 0:
#                     scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len)
#                     targets = caps_sorted[:, 1:]
#                     scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
#                     targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
#                     cap_loss = self.criterion_cap(scores, targets.to(torch.int64))
#
#                 if self.current_goal == 0:
#                     loss = det_loss
#                 elif self.current_goal == 1:
#                     loss = cap_loss
#                 else:
#                     if args.use_loss_balance == 1:
#                         eps = 1e-8
#                         det_den = det_loss.detach()
#                         cap_den = cap_loss.detach() if cap_loss is not None else torch.tensor(1.0, device=det_loss.device)
#                         loss = det_loss / (det_den + eps) + (cap_loss / (cap_den + eps) if cap_loss is not None else 0.0)
#                     else:
#                         loss = det_loss + (cap_loss if cap_loss is not None else 0.0)
#
#                 loss = loss / accum_steps
#
#             self.scaler.scale(loss).backward()
#
#             boundary = ((i + 1) % accum_steps == 0) or ((i + 1) == len(self.train_loader))
#             if boundary:
#                 self.unscale_and_clip()
#                 if self.decoder_optimizer is not None:
#                     self.scaler.step(self.decoder_optimizer)
#                 self.scaler.step(self.encoder_trans_optimizer)
#                 if self.encoder_optimizer is not None:
#                     self.scaler.step(self.encoder_optimizer)
#                 self.scaler.update()
#                 self.zero_all()
#
#             running_det += float(det_loss.item())
#             if cap_loss is not None:
#                 running_cap += float(cap_loss.item())
#             running_joint += float(loss.item()) * accum_steps
#             num_batches += 1
#
#             self.hist[self.index_i, 0] = time.time() - t0
#             self.hist[self.index_i, 1] = float(det_loss.item())
#             self.hist[self.index_i, 2] = accuracy(
#                 seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
#                 seg_label.reshape(-1), 1
#             )
#             if cap_loss is not None:
#                 self.hist[self.index_i, 3] = float(cap_loss.item())
#                 self.hist[self.index_i, 4] = accuracy(scores, targets, 5)
#             self.index_i += 1
#
#             if self.index_i % args.print_freq == 0:
#                 w0, w1 = max(0, self.index_i - args.print_freq), self.index_i
#                 print_log(
#                     'Training Epoch: [{0}][{1}/{2}]  '
#                     'Avg Batch Time: {3:.3f}  Det_Loss: {4:.4f}  Det Acc: {5:.3f}  '
#                     'Cap_loss: {6:.5f}  Text_Top-5 Acc: {7:.3f}'
#                     .format(
#                         epoch, i, len(self.train_loader),
#                         np.mean(self.hist[w0:w1, 0]),
#                         np.mean(self.hist[w0:w1, 1]),
#                         np.mean(self.hist[w0:w1, 2]),
#                         np.mean(self.hist[w0:w1, 3]),
#                         np.mean(self.hist[w0:w1, 4])
#                     ), self.log
#                 )
#
#         avg_det = running_det / max(1, num_batches)
#         avg_cap = running_cap / max(1, num_batches)
#         avg_joint = running_joint / max(1, num_batches)
#         print_log(f'Epoch {epoch} Avg - Det Loss: {avg_det:.4f}, Cap Loss: {avg_cap:.4f}, Joint Loss: {avg_joint:.4f}', self.log)
#
#     def validation(self, epoch):
#         word_vocab = self.word_vocab
#         # Respect partial-unfreeze mode: do not override encoder_trans to eval if managed
#         self.decoder.eval()
#         if not self.manage_encoder_trans_mode:
#             self.encoder_trans.eval()
#         self.encoder.eval()
#
#         val_start_time = time.time()
#         references, hypotheses = [], []
#         self.evaluator.reset()
#
#         with torch.no_grad():
#             for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _) in enumerate(
#                     tqdm(self.val_loader, desc=f'val_BEAM {self.args.beam_size}')):
#                 imgA = imgA.cuda(non_blocking=True)
#                 imgB = imgB.cuda(non_blocking=True)
#                 seg_label = seg_label.cuda(non_blocking=True)
#                 token_all = token_all.squeeze(0).cuda(non_blocking=True)  # val batchsize=1
#
#                 feat1_list, feat2_list = self.encoder(imgA, imgB)
#                 feat1, feat2, seg_pre = self.encoder_trans(feat1_list, feat2_list)
#
#                 if self.current_goal != 0:
#                     if self.args.beam_size > 1:
#                         seq = self.decoder.sample_beam(feat1, feat2, k=self.args.beam_size)
#                     else:
#                         seq = self.decoder.sample(feat1, feat2, k=1)
#
#                 if self.current_goal != 1:
#                     pred_seg = seg_pre.data.cpu().numpy()
#                     seg_label_np = seg_label.cpu().numpy()
#                     pred_seg = np.argmax(pred_seg, axis=1)
#                     self.evaluator.add_batch(seg_label_np, pred_seg)
#
#                 if self.current_goal != 0:
#                     img_token = token_all.tolist()
#                     img_tokens = list(map(lambda c: [w for w in c if w not in {
#                         word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
#                     references.append(img_tokens)
#                     pred_seq = [w for w in seq if w not in {
#                         word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
#                     hypotheses.append(pred_seq)
#                     assert len(references) == len(hypotheses)
#
#         val_time = time.time() - val_start_time
#
#         Acc_seg = Acc_class_seg = mIoU_seg = FWIoU_seg = 0.0
#         if self.current_goal != 1:
#             Acc_seg = self.evaluator.Pixel_Accuracy()
#             Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
#             mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
#             FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
#             print_log(
#                 '\nDetection_Validation:\n'
#                 f'Acc_seg: {Acc_seg:.5f}  Acc_class_seg: {Acc_class_seg:.5f}  '
#                 f'mIoU_seg: {mIoU_seg:.5f}  FWIoU_seg: {FWIoU_seg:.5f}', self.log
#             )
#             print_log(f'Iou: {IoU}', self.log)
#
#         Bleu_1 = Bleu_2 = Bleu_3 = Bleu_4 = Meteor = Rouge = Cider = 0.0
#         if self.current_goal != 0:
#             score_dict = get_eval_score(references, hypotheses)
#             Bleu_1 = score_dict['Bleu_1']; Bleu_2 = score_dict['Bleu_2']
#             Bleu_3 = score_dict['Bleu_3']; Bleu_4 = score_dict['Bleu_4']
#             Meteor = score_dict['METEOR']; Rouge = score_dict['ROUGE_L']; Cider = score_dict['CIDEr']
#             print_log('Captioning_Validation:\n'
#                       f'Time: {val_time:.3f}  BLEU-1: {Bleu_1:.5f}  BLEU-2: {Bleu_2:.5f}  BLEU-3: {Bleu_3:.5f}  '
#                       f'BLEU-4: {Bleu_4:.5f}  Meteor: {Meteor:.5f}  Rouge: {Rouge:.5f}  Cider: {Cider:.5f}', self.log)
#
#         metric_value = metric_for_goal(self.current_goal, Bleu_4, mIoU_seg)
#         if self.encoder_lr_scheduler is not None:
#             self.encoder_lr_scheduler.step(metric_value)
#         self.encoder_trans_lr_scheduler.step(metric_value)
#         if self.decoder_lr_scheduler is not None:
#             self.decoder_lr_scheduler.step(metric_value)
#
#         # Save checkpoint if improved
#         improved = False
#         if self.current_goal == 2:
#             improved = (Bleu_4 + mIoU_seg) >= self.Sum_Metric
#             if improved:
#                 self.Sum_Metric = Bleu_4 + mIoU_seg
#         elif self.current_goal == 1:
#             improved = Bleu_4 >= self.best_bleu4
#             if improved:
#                 self.best_bleu4 = Bleu_4
#         else:
#             improved = mIoU_seg >= self.MIou
#             if improved:
#                 self.MIou = mIoU_seg
#
#         if improved:
#             self.no_improve_epochs = 0
#             print('Save Model')
#             state = {
#                 'encoder_dict': self.encoder.state_dict(),
#                 'encoder_trans_dict': self.encoder_trans.state_dict(),
#                 'decoder_dict': self.decoder.state_dict()
#             }
#
#             metric = f'Sum_{round(100000 * (Bleu_4 + mIoU_seg))}_MIou_{round(100000 * mIoU_seg)}_Bleu4_{round(100000 * Bleu_4)}'
#             preferred_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
#             fallback_name = f'{self.args.data_name}_epo_{epoch}.pth'
#
#             best_model_path = safe_torch_save(
#                 state, self.args.savepath, preferred_name, fallback_name,
#                 log_fn=lambda m: print_log(m, self.log)
#             )
#             self.best_epoch = epoch
#             self.best_model_path = best_model_path
#         else:
#             self.no_improve_epochs += 1
#
#         enc_lr = self.encoder_optimizer.param_groups[0]['lr'] if self.encoder_optimizer is not None else 0.0
#         enc_tr_lr = self.encoder_trans_optimizer.param_groups[0]['lr']
#         dec_lr = self.decoder_optimizer.param_groups[0]['lr'] if self.decoder_optimizer is not None else 0.0
#         print_log(f'LRs — encoder: {enc_lr:.6f}, encoder_trans: {enc_tr_lr:.6f}, decoder: {dec_lr:.6f}', self.log)
#
#     def fit(self):
#         for epoch in range(self.args.start_epoch, self.args.num_epochs):
#             self.training(epoch)
#             if (epoch % self.args.val_interval == 0) or (epoch == self.args.num_epochs - 1):
#                 self.validation(epoch)
#             if self.no_improve_epochs >= self.args.early_stop_patience:
#                 print_log(f'Early stopping at epoch {epoch} (no improvement for {self.args.early_stop_patience} epochs)', self.log)
#                 break
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')
#
#     # Data
#     parser.add_argument('--data_folder', default=r'C:\Users\nemade\Downloads\LEVIR-MCI-data\LEVIR-MCI-dataset\images')
#     parser.add_argument('--list_path', default='./data/LEVIR_MCI/')
#     parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/')
#     parser.add_argument('--vocab_file', default='vocab')
#     parser.add_argument('--source_vocab_json', type=str, default='', help='path to original vocab (token->idx) for transplant')
#     parser.add_argument('--max_length', type=int, default=46)
#     parser.add_argument('--allow_unk', type=int, default=1)
#     parser.add_argument('--data_name', default='LEVIR_MCI')
#
#     # Runtime
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--print_freq', type=int, default=100)
#     parser.add_argument('--val_interval', type=int, default=1)
#     parser.add_argument('--start_epoch', type=int, default=0)
#     parser.add_argument('--savepath', default='./models_ckpt/')
#     parser.add_argument('--beam_size', type=int, default=1, help='validation beam size (set to 3 for BLEU fairness; val batch must be 1)')
#     parser.add_argument('--use_compile', type=bool, default=False)
#
#     # Stage/goal/checkpoint
#     parser.add_argument('--train_goal', type=int, default=2, help='0: det; 1: cap; 2: joint')
#     parser.add_argument('--train_stage', default='s2', help='s1: pretrain; s2: finetune from checkpoint')
#     parser.add_argument('--checkpoint', required=True, help='path to checkpoint used for s2')
#
#     # Performance
#     parser.add_argument('--num_workers', type=int, default=0)
#
#     # LRs and clipping
#     parser.add_argument('--encoder_lr', type=float, default=1e-4)
#     parser.add_argument('--decoder_lr', type=float, default=2e-4)
#     parser.add_argument('--grad_clip', type=float, default=1.0)
#
#     # Schedulers
#     parser.add_argument('--lr_factor', type=float, default=0.5)
#     parser.add_argument('--lr_patience', type=int, default=5)
#     parser.add_argument('--early_stop_patience', type=int, default=20)
#
#     # Model params
#     parser.add_argument('--train_batchsize', type=int, default=32)
#     parser.add_argument('--val_batchsize', type=int, default=1)
#     parser.add_argument('--num_epochs', type=int, default=50)
#     parser.add_argument('--dropout', type=float, default=0.1)
#     parser.add_argument('--network', default='segformer-mit_b1')
#     parser.add_argument('--encoder_dim', type=int, default=512)
#     parser.add_argument('--feat_size', type=int, default=16)
#     parser.add_argument('--n_heads', type=int, default=8)
#     parser.add_argument('--n_layers', type=int, default=3)
#     parser.add_argument('--decoder_n_layers', type=int, default=1)
#     parser.add_argument('--feature_dim', type=int, default=512)
#
#     # New flags (defaults preserve your current behavior)
#     parser.add_argument('--cap_unfreeze_encoder_trans', type=int, default=0,
#                         help='if 1 and goal==1: unfreeze only caption modules in encoder_trans; default 0 keeps it fully frozen')
#     parser.add_argument('--use_loss_balance', type=int, default=0,
#                         help='if 1 and goal==2: normalize det and cap losses before summation')
#
#     args = parser.parse_args()
#     torch.cuda.set_device(args.gpu_id)
#     trainer = Trainer(args)
#     print('Starting Epoch:', args.start_epoch)
#     print('Total Epochs:', args.num_epochs)
#     trainer.fit()





###############################  added label smoothing and weight decay ###############
"""
error-free train_newoptimized.py that adds:

    cap_unfreeze_encoder_trans flag (default 0 = unchanged behavior).
    use_loss_balance flag for joint normalization (default 0 = OFF).
    cap_label_smoothing (default 0.0).
    Per-optimizer weight decay: encoder_wd, encoder_trans_wd, decoder_wd (all default 0.0).
    Sanity prints: trainable parameter counts and the mode chosen.

"""
# train_newoptimized.py
import argparse
import json
import os
import time
import random
import numpy as np
import torch
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

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


def metric_for_goal(train_goal, bleu4, miou):
    if train_goal == 1:
        return bleu4
    if train_goal == 0:
        return miou
    return bleu4 + miou


def transplant_decoder_vocab_rows(decoder_module, checkpoint_decoder_sd, source_vocab_path, target_vocab_dict, log_fn=print):
    if not source_vocab_path or not os.path.exists(source_vocab_path):
        log_fn("Vocab transplant: source_vocab_json not found; skipping.")
        return 0, 0

    with open(source_vocab_path, 'r') as f:
        src_tok2idx = json.load(f)
    tgt_tok2idx = target_vocab_dict

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

#########Commenting the below because new updated code ######
# def safe_torch_save(state, dir_path, preferred_filename, fallback_filename, log_fn=print):
#     os.makedirs(dir_path, exist_ok=True)
#     full_pref = os.path.join(dir_path, preferred_filename)
#     try:
#         torch.save(state, full_pref)
#         return full_pref
#     except Exception as e:
#         log_fn(f"Warning: save failed for '{full_pref}'. Retrying with compact name. Error: {e}")
#         full_fallback = os.path.join(dir_path, fallback_filename)
#         torch.save(state, full_fallback)
#         return full_fallback

def safe_torch_save(state, dir_path, preferred_filename, fallback_filename, log_fn=print):
    os.makedirs(dir_path, exist_ok=True)
    tmp = os.path.join(dir_path, f'~tmp_{int(time.time())}.pth')
    full_pref = os.path.join(dir_path, preferred_filename)
    try:
        torch.save(state, tmp, _use_new_zipfile_serialization=False)
        os.replace(tmp, full_pref)
        return full_pref
    except Exception as e1:
        log_fn(f"Warning: save failed for '{full_pref}' ({e1}). Retrying fallback...")
        try:
            full_fb = os.path.join(dir_path, fallback_filename)
            torch.save(state, tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, full_fb)
            return full_fb
        except Exception as e2:
            log_fn(f"[FATAL] Fallback save failed: {e2}. Skipping save this epoch.")
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except Exception:
                pass
            return None



def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.current_goal = args.train_goal

        run_name = f'baseline_{time_file_str()}_train_goal_{args.train_goal}_{random.randint(10, 100)}'
        self.args.savepath = os.path.join(args.savepath, run_name)
        os.makedirs(self.args.savepath, exist_ok=True)
        self.log = open(os.path.join(self.args.savepath, f'{run_name}.log'), 'w')

        print_log(f'=>datset: {args.data_name}', self.log)
        print_log(f'=>network: {args.network}', self.log)
        print_log(f'=>encoder_lr: {args.encoder_lr}', self.log)
        print_log(f'=>decoder_lr: {args.decoder_lr}', self.log)
        print_log(f'=>num_epochs: {args.num_epochs}', self.log)
        print_log(f'=>train_batchsize: {args.train_batchsize}', self.log)
        print_log(f'=>loss_balance: {"ON" if (args.use_loss_balance and args.train_goal==2) else "OFF"}', self.log)
        print_log(f'=>cap_label_smoothing: {args.cap_label_smoothing}', self.log)
        print_log(f'=>wd enc/enc_trans/dec: {args.encoder_wd} / {args.encoder_trans_wd} / {args.decoder_wd}', self.log)
        print_log(f'=>cap_unfreeze_encoder_trans: {args.cap_unfreeze_encoder_trans}', self.log)

        with open(os.path.join(args.list_path + args.vocab_file + '.json'), 'r') as f:
            self.word_vocab = json.load(f)

        self.build_model()

        self.criterion_cap = torch.nn.CrossEntropyLoss(label_smoothing=self.args.cap_label_smoothing).cuda()
        self.criterion_det = torch.nn.CrossEntropyLoss().cuda()
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        if args.data_name == 'LEVIR_MCI':
            self.train_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'train',
                               args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.train_batchsize, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                persistent_workers=(args.num_workers > 0)
            )
            self.val_loader = data.DataLoader(
                LEVIRCCDataset(args.data_folder, args.list_path, 'val',
                               args.token_folder, args.vocab_file, args.max_length, args.allow_unk),
                batch_size=args.val_batchsize, shuffle=False,
                num_workers=0, pin_memory=True
            )

        self.index_i = 0
        self.hist = np.zeros((max(1, args.num_epochs) * max(1, len(self.train_loader)), 5))
        self.best_bleu4 = 0.0
        self.MIou = 0.0
        self.Sum_Metric = 0.0
        self.best_epoch = -1
        self.best_model_path = None
        self.no_improve_epochs = 0

        self.evaluator = Evaluator(num_class=3)

    def build_model(self):
        args = self.args
        self.encoder = Encoder(args.network)
        self.encoder_trans = AttentiveEncoder(
            args.train_stage, args.n_layers,
            (args.feat_size, args.feat_size, args.encoder_dim),
            args.n_heads
        )
        self.decoder = DecoderTransformer(
            encoder_dim=args.encoder_dim,
            feature_dim=args.feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=args.max_length,
            word_vocab=self.word_vocab,
            n_head=args.n_heads,
            n_layers=args.decoder_n_layers,
            dropout=args.dropout
        )

        self.manage_encoder_trans_mode = False
        fine_tune_cap_decoder = True

        if args.train_stage == 's1':
            args.fine_tune_encoder = True
            self.encoder.fine_tune(args.fine_tune_encoder)
            self.encoder_trans.fine_tune(self.current_goal)
            self.manage_encoder_trans_mode = True
            self.decoder.fine_tune(fine_tune_cap_decoder)

        elif args.train_stage == 's2':
            if args.checkpoint is None or not os.path.exists(args.checkpoint):
                raise ValueError(f'Error: checkpoint must be set for train_stage=s2. Got: {args.checkpoint}')

            checkpoint = torch.load(args.checkpoint, map_location='cuda')
            print_log(f'Load Model from {args.checkpoint}', self.log)

            self.encoder.load_state_dict(checkpoint['encoder_dict'], strict=True)
            self.encoder_trans.load_state_dict(checkpoint['encoder_trans_dict'], strict=False)

            dec_sd_ckpt = checkpoint['decoder_dict']
            model_sd = self.decoder.state_dict()
            skip = 0
            for k, v in dec_sd_ckpt.items():
                if k in model_sd and model_sd[k].shape == v.shape and not k.startswith('vocab_embedding') and not k.startswith('wdc.'):
                    model_sd[k].copy_(v)
                elif k in model_sd and model_sd[k].shape == v.shape:
                    model_sd[k].copy_(v)
                else:
                    skip += 1
            self.decoder.load_state_dict(model_sd, strict=False)
            if skip > 0:
                print_log(f'Note: skipped {skip} decoder params due to shape mismatch (likely vocab change).', self.log)

            old_vocab_size = dec_sd_ckpt.get('wdc.weight', torch.empty(0)).shape[0] if 'wdc.weight' in dec_sd_ckpt else None
            new_vocab_size = self.decoder.wdc.weight.shape[0]
            if args.source_vocab_json and os.path.exists(args.source_vocab_json) and old_vocab_size and new_vocab_size != old_vocab_size:
                transplant_decoder_vocab_rows(
                    decoder_module=self.decoder,
                    checkpoint_decoder_sd=dec_sd_ckpt,
                    source_vocab_path=args.source_vocab_json,
                    target_vocab_dict=self.word_vocab,
                    log_fn=lambda m: print_log(m, self.log)
                )
            else:
                print_log("Vocab transplant: skipped (no source_vocab_json provided or vocab sizes match).", self.log)

            if args.train_goal == 2:
                args.fine_tune_encoder = True
                self.encoder.fine_tune(True)
                for p in self.encoder_trans.parameters():
                    p.requires_grad = True
                fine_tune_cap_decoder = True

            elif args.train_goal == 1:
                args.fine_tune_encoder = False
                self.encoder.fine_tune(False)
                if args.cap_unfreeze_encoder_trans == 1:
                    self.encoder_trans.fine_tune(1)
                    self.manage_encoder_trans_mode = True
                else:
                    for p in self.encoder_trans.parameters():
                        p.requires_grad = False
                fine_tune_cap_decoder = True

            else:  # goal == 0
                args.fine_tune_encoder = True
                self.encoder.fine_tune(True)
                for p in self.encoder_trans.parameters():
                    p.requires_grad = True
                fine_tune_cap_decoder = False

            self.decoder.fine_tune(fine_tune_cap_decoder)
        else:
            raise ValueError('train_stage must be s1 or s2.')

        self.encoder = self.encoder.cuda()
        self.encoder_trans = self.encoder_trans.cuda()
        self.decoder = self.decoder.cuda()

        if getattr(args, 'use_compile', False):
            self.encoder_trans = torch.compile(self.encoder_trans)
            self.decoder = torch.compile(self.decoder)

        self.encoder_optimizer = torch.optim.Adam(
            params=self.encoder.parameters(), lr=args.encoder_lr, weight_decay=args.encoder_wd
        ) if args.fine_tune_encoder else None

        # self.encoder_trans_optimizer = torch.optim.Adam(
        #     params=filter(lambda p: p.requires_grad, self.encoder_trans.parameters()),
        #     lr=args.encoder_lr, weight_decay=args.encoder_trans_wd
        # )
        #### Creating encoder_trans optimizer only if there are trainable params

        et_params = [p for p in self.encoder_trans.parameters() if p.requires_grad]
        self.encoder_trans_optimizer = (
            torch.optim.Adam(et_params, lr=args.encoder_lr, weight_decay=args.encoder_trans_wd)
            if len(et_params) > 0 else None
        )

        self.decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, self.decoder.parameters()),
            lr=args.decoder_lr, weight_decay=args.decoder_wd
        ) if fine_tune_cap_decoder else None

        from torch.optim.lr_scheduler import ReduceLROnPlateau
        factor = args.lr_factor
        patience = args.lr_patience
        self.encoder_lr_scheduler = ReduceLROnPlateau(
            self.encoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True
        ) if self.encoder_optimizer is not None else None

        # self.encoder_trans_lr_scheduler = ReduceLROnPlateau(
        #     self.encoder_trans_optimizer, mode='max', factor=factor, patience=patience, verbose=True
        # )
        ##### Create scheduler only if optimizer exists

        self.encoder_trans_lr_scheduler = (
            ReduceLROnPlateau(self.encoder_trans_optimizer, mode='max', factor=factor, patience=patience, verbose=True)
            if self.encoder_trans_optimizer is not None else None
        )
        self.decoder_lr_scheduler = ReduceLROnPlateau(
            self.decoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True
        ) if self.decoder_optimizer is not None else None

        # Sanity: print trainable params
        et_total, et_train = count_params(self.encoder_trans)
        dec_total, dec_train = count_params(self.decoder)
        enc_total, enc_train = count_params(self.encoder)
        print_log(f'Params — encoder: {enc_train}/{enc_total}, encoder_trans: {et_train}/{et_total}, decoder: {dec_train}/{dec_total}', self.log)
        if self.current_goal == 1 and self.args.cap_unfreeze_encoder_trans == 1 and et_train == 0:
            print_log('WARNING: encoder_trans seems frozen in goal=1 though cap_unfreeze_encoder_trans=1 was set.', self.log)

    def set_modes(self):
        g = self.current_goal
        if g == 2:
            self.encoder.train()
            if not self.manage_encoder_trans_mode:
                self.encoder_trans.train()
            self.decoder.train()
        elif g == 1:
            self.encoder.eval()
            if not self.manage_encoder_trans_mode:
                self.encoder_trans.eval()
            self.decoder.train()
        else:
            self.encoder.train()
            if not self.manage_encoder_trans_mode:
                self.encoder_trans.train()
            self.decoder.eval()

    def zero_all(self):
        if self.decoder_optimizer is not None:
            self.decoder_optimizer.zero_grad(set_to_none=True)
        # self.encoder_trans_optimizer.zero_grad(set_to_none=True) #Guard uses in zero_all
        if self.encoder_trans_optimizer is not None:
            self.encoder_trans_optimizer.zero_grad(set_to_none=True)
        if self.encoder_optimizer is not None:
            self.encoder_optimizer.zero_grad(set_to_none=True)

    def unscale_and_clip(self):
        max_norm = self.args.grad_clip
        if max_norm is None:
            return
        if self.decoder_optimizer is not None:
            self.scaler.unscale_(self.decoder_optimizer)
        # self.scaler.unscale_(self.encoder_trans_optimizer)
        if self.encoder_trans_optimizer is not None:
            self.scaler.unscale_(self.encoder_trans_optimizer)
        if self.encoder_optimizer is not None:
            self.scaler.unscale_(self.encoder_optimizer)
        if self.decoder_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm)
        torch.nn.utils.clip_grad_norm_(self.encoder_trans.parameters(), max_norm)
        if self.encoder_optimizer is not None:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm)

    def training(self, epoch):
        args = self.args
        self.set_modes()
        self.zero_all()

        running_det = 0.0
        running_cap = 0.0
        running_joint = 0.0
        num_batches = 0

        accum_steps = max(1, 64 // max(1, args.train_batchsize))

        for i, (imgA, imgB, seg_label, _, _, token, token_len, _) in enumerate(self.train_loader):
            t0 = time.time()

            imgA = imgA.cuda(non_blocking=True)
            imgB = imgB.cuda(non_blocking=True)
            seg_label = seg_label.cuda(non_blocking=True)
            token = token.squeeze(1).cuda(non_blocking=True)
            token_len = token_len.cuda(non_blocking=True)

            with autocast(enabled=torch.cuda.is_available()):
                feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)

                det_loss = self.criterion_det(seg_pre, seg_label.to(torch.int64))

                cap_loss = None
                if self.current_goal != 0:
                    scores, caps_sorted, decode_lengths, sort_ind = self.decoder(feat1, feat2, token, token_len)
                    targets = caps_sorted[:, 1:]
                    scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
                    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                    cap_loss = self.criterion_cap(scores, targets.to(torch.int64))

                if self.current_goal == 0:
                    loss = det_loss
                elif self.current_goal == 1:
                    loss = cap_loss
                else:
                    if args.use_loss_balance == 1:
                        eps = 1e-8
                        det_den = det_loss.detach()
                        cap_den = cap_loss.detach() if cap_loss is not None else torch.tensor(1.0, device=det_loss.device)
                        loss = det_loss / (det_den + eps) + (cap_loss / (cap_den + eps) if cap_loss is not None else 0.0)
                    else:
                        loss = det_loss + (cap_loss if cap_loss is not None else 0.0)

                loss = loss / accum_steps

            self.scaler.scale(loss).backward()

            boundary = ((i + 1) % accum_steps == 0) or ((i + 1) == len(self.train_loader))
            if boundary:
                self.unscale_and_clip()
                if self.decoder_optimizer is not None:
                    self.scaler.step(self.decoder_optimizer)
                # self.scaler.step(self.encoder_trans_optimizer)
                if self.encoder_trans_optimizer is not None:
                    self.scaler.step(self.encoder_trans_optimizer)
                if self.encoder_optimizer is not None:
                    self.scaler.step(self.encoder_optimizer)
                self.scaler.update()
                self.zero_all()

            running_det += float(det_loss.item())
            if cap_loss is not None:
                running_cap += float(cap_loss.item())
            running_joint += float(loss.item()) * accum_steps
            num_batches += 1

            self.hist[self.index_i, 0] = time.time() - t0
            self.hist[self.index_i, 1] = float(det_loss.item())
            self.hist[self.index_i, 2] = accuracy(
                seg_pre.permute(0, 2, 3, 1).reshape(-1, seg_pre.size(1)),
                seg_label.reshape(-1), 1
            )
            if cap_loss is not None:
                self.hist[self.index_i, 3] = float(cap_loss.item())
                self.hist[self.index_i, 4] = accuracy(scores, targets, 5)
            self.index_i += 1

            if self.index_i % args.print_freq == 0:
                w0, w1 = max(0, self.index_i - args.print_freq), self.index_i
                print_log(
                    'Training Epoch: [{0}][{1}/{2}]  '
                    'Avg Batch Time: {3:.3f}  Det_Loss: {4:.4f}  Det Acc: {5:.3f}  '
                    'Cap_loss: {6:.5f}  Text_Top-5 Acc: {7:.3f}'
                    .format(
                        epoch, i, len(self.train_loader),
                        np.mean(self.hist[w0:w1, 0]),
                        np.mean(self.hist[w0:w1, 1]),
                        np.mean(self.hist[w0:w1, 2]),
                        np.mean(self.hist[w0:w1, 3]),
                        np.mean(self.hist[w0:w1, 4])
                    ), self.log
                )

        avg_det = running_det / max(1, num_batches)
        avg_cap = running_cap / max(1, num_batches)
        avg_joint = running_joint / max(1, num_batches)
        print_log(f'Epoch {epoch} Avg - Det Loss: {avg_det:.4f}, Cap Loss: {avg_cap:.4f}, Joint Loss: {avg_joint:.4f}', self.log)

    def validation(self, epoch):
        word_vocab = self.word_vocab
        self.decoder.eval()
        if not self.manage_encoder_trans_mode:
            self.encoder_trans.eval()
        self.encoder.eval()

        val_start_time = time.time()
        references, hypotheses = [], []
        self.evaluator.reset()

        with torch.no_grad():
            for ind, (imgA, imgB, seg_label, token_all, token_all_len, _, _, _) in enumerate(
                    tqdm(self.val_loader, desc=f'val_BEAM {self.args.beam_size}')):
                imgA = imgA.cuda(non_blocking=True)
                imgB = imgB.cuda(non_blocking=True)
                token_all = token_all.squeeze(0).cuda(non_blocking=True)

                feat1, feat2 = self.encoder(imgA, imgB)
                feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)

                if self.current_goal != 0:
                    if self.args.beam_size > 1:
                        seq = self.decoder.sample_beam(feat1, feat2, k=self.args.beam_size)
                    else:
                        seq = self.decoder.sample(feat1, feat2, k=1)

                if self.current_goal != 1:
                    pred_seg = seg_pre.data.cpu().numpy()
                    seg_label_np = seg_label.cpu().numpy()
                    pred_seg = np.argmax(pred_seg, axis=1)
                    self.evaluator.add_batch(seg_label_np, pred_seg)

                if self.current_goal != 0:
                    img_token = token_all.tolist()
                    img_tokens = list(map(lambda c: [w for w in c if w not in {
                        word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}], img_token))
                    references.append(img_tokens)
                    pred_seq = [w for w in seq if w not in {
                        word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}]
                    hypotheses.append(pred_seq)
                    assert len(references) == len(hypotheses)

        val_time = time.time() - val_start_time

        Acc_seg = Acc_class_seg = mIoU_seg = FWIoU_seg = 0.0
        if self.current_goal != 1:
            Acc_seg = self.evaluator.Pixel_Accuracy()
            Acc_class_seg = self.evaluator.Pixel_Accuracy_Class()
            mIoU_seg, IoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU_seg = self.evaluator.Frequency_Weighted_Intersection_over_Union()
            print_log(
                '\nDetection_Validation:\n'
                f'Acc_seg: {Acc_seg:.5f}  Acc_class_seg: {Acc_class_seg:.5f}  '
                f'mIoU_seg: {mIoU_seg:.5f}  FWIoU_seg: {FWIoU_seg:.5f}', self.log
            )
            print_log(f'Iou: {IoU}', self.log)

        Bleu_1 = Bleu_2 = Bleu_3 = Bleu_4 = Meteor = Rouge = Cider = 0.0
        if self.current_goal != 0:
            score_dict = get_eval_score(references, hypotheses)
            Bleu_1 = score_dict['Bleu_1']; Bleu_2 = score_dict['Bleu_2']
            Bleu_3 = score_dict['Bleu_3']; Bleu_4 = score_dict['Bleu_4']
            Meteor = score_dict['METEOR']; Rouge = score_dict['ROUGE_L']; Cider = score_dict['CIDEr']
            print_log('Captioning_Validation:\n'
                      f'Time: {val_time:.3f}  BLEU-1: {Bleu_1:.5f}  BLEU-2: {Bleu_2:.5f}  BLEU-3: {Bleu_3:.5f}  '
                      f'BLEU-4: {Bleu_4:.5f}  Meteor: {Meteor:.5f}  Rouge: {Rouge:.5f}  Cider: {Cider:.5f}', self.log)

        metric_value = metric_for_goal(self.current_goal, Bleu_4, mIoU_seg)
        if self.encoder_lr_scheduler is not None:
            self.encoder_lr_scheduler.step(metric_value)
        # self.encoder_trans_lr_scheduler.step(metric_value) #gaurd
        if self.encoder_trans_lr_scheduler is not None:
            self.encoder_trans_lr_scheduler.step(metric_value)
        if self.decoder_lr_scheduler is not None:
            self.decoder_lr_scheduler.step(metric_value)

        improved = False
        if self.current_goal == 2:
            improved = (Bleu_4 + mIoU_seg) >= self.Sum_Metric
            if improved:
                self.Sum_Metric = Bleu_4 + mIoU_seg
        elif self.current_goal == 1:
            improved = Bleu_4 >= self.best_bleu4
            if improved:
                self.best_bleu4 = Bleu_4
        else:
            improved = mIoU_seg >= self.MIou
            if improved:
                self.MIou = mIoU_seg

        if improved:
            self.no_improve_epochs = 0
            print('Save Model')
            state = {
                'encoder_dict': self.encoder.state_dict(),
                'encoder_trans_dict': self.encoder_trans.state_dict(),
                'decoder_dict': self.decoder.state_dict()
            }

            metric = f'Sum_{round(100000 * (Bleu_4 + mIoU_seg))}_MIou_{round(100000 * mIoU_seg)}_Bleu4_{round(100000 * Bleu_4)}'
            preferred_name = f'{self.args.data_name}_bts_{self.args.train_batchsize}_{self.args.network}_epo_{epoch}_{metric}.pth'
            fallback_name = f'{self.args.data_name}_epo_{epoch}.pth'

            best_model_path = safe_torch_save(
                state, self.args.savepath, preferred_name, fallback_name,
                log_fn=lambda m: print_log(m, self.log)
            )
            self.best_epoch = epoch
            self.best_model_path = best_model_path
        else:
            self.no_improve_epochs += 1

        enc_lr = self.encoder_optimizer.param_groups[0]['lr'] if self.encoder_optimizer is not None else 0.0
        # enc_tr_lr = self.encoder_trans_optimizer.param_groups[0]['lr']
        enc_tr_lr = (self.encoder_trans_optimizer.param_groups[0]['lr']
                     if self.encoder_trans_optimizer is not None else 0.0)
        dec_lr = self.decoder_optimizer.param_groups[0]['lr'] if self.decoder_optimizer is not None else 0.0
        print_log(f'LRs — encoder: {enc_lr:.6f}, encoder_trans: {enc_tr_lr:.6f}, decoder: {dec_lr:.6f}', self.log)

    def fit(self):
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.training(epoch)
            if (epoch % self.args.val_interval == 0) or (epoch == self.args.num_epochs - 1):
                self.validation(epoch)
            if self.no_improve_epochs >= self.args.early_stop_patience:
                print_log(f'Early stopping at epoch {epoch} (no improvement for {self.args.early_stop_patience} epochs)', self.log)
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote_Sensing_Image_Change_Interpretation')

    # Data
    parser.add_argument('--data_folder', default=r'C:\Users\nemade\Downloads\LEVIR-MCI-data\LEVIR-MCI-dataset\images', help='folder with data files')
    parser.add_argument('--list_path', default='./data/LEVIR_MCI/')
    parser.add_argument('--token_folder', default='./data/LEVIR_MCI/tokens/')
    parser.add_argument('--vocab_file', default='vocab')
    parser.add_argument('--source_vocab_json', type=str, default='', help='path to original vocab (token->idx) for transplant')
    parser.add_argument('--max_length', type=int, default=46)
    parser.add_argument('--allow_unk', type=int, default=1)
    parser.add_argument('--data_name', default='LEVIR_MCI')

    # Runtime
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--savepath', default='./models_ckpt/')
    parser.add_argument('--beam_size', type=int, default=1, help='validation beam size (val batch must be 1)')
    parser.add_argument('--short_save_names', type=bool, default=False)
    parser.add_argument('--use_compile', type=bool, default=False)

    # Stage/goal/checkpoint
    parser.add_argument('--train_goal', type=int, default=2, help='0: det; 1: cap; 2: joint')
    parser.add_argument('--train_stage', default='s2', help='s1: pretrain; s2: finetune from checkpoint')
    # parser.add_argument('--checkpoint', required=True, help='path to checkpoint used for s2')
    parser.add_argument('--checkpoint', default='', help='path to checkpoint used for s2')


    # Performance
    parser.add_argument('--num_workers', type=int, default=0)

    # LRs and clipping
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)

    # Weight decay (defaults keep old behavior)
    parser.add_argument('--encoder_wd', type=float, default=0.0)
    parser.add_argument('--encoder_trans_wd', type=float, default=0.0)
    parser.add_argument('--decoder_wd', type=float, default=0.0)

    # Schedulers
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=5)
    parser.add_argument('--early_stop_patience', type=int, default=20)

    # Model params
    parser.add_argument('--train_batchsize', type=int, default=32)
    parser.add_argument('--val_batchsize', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--network', default='segformer-mit_b1')
    parser.add_argument('--encoder_dim', type=int, default=512)
    parser.add_argument('--feat_size', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--decoder_n_layers', type=int, default=1)
    parser.add_argument('--feature_dim', type=int, default=512)

    # New flags
    parser.add_argument('--cap_unfreeze_encoder_trans', type=int, default=0,
                        help='if 1 and goal==1: unfreeze only caption modules in encoder_trans; default 0 keeps it frozen')
    parser.add_argument('--use_loss_balance', type=int, default=0,
                        help='if 1 and goal==2: normalize det and cap losses before summation')
    parser.add_argument('--cap_label_smoothing', type=float, default=0.0,
                        help='label smoothing for caption CE loss (0.0 keeps old behavior)')
    # train_newoptimized.py: after args parsing
    parser.add_argument('--eval_only', type=int, default=0)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    trainer = Trainer(args)
    # after trainer = Trainer(args)
    if args.eval_only == 1:
        trainer.validation(epoch=0)
        exit(0)
    print('Starting Epoch:', args.start_epoch) 
    print('Total Epochs:', args.num_epochs)
    trainer.fit()
