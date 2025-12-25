#Previous LEVIR_MVVI.py gave an error that could not broadcast input array from shape (46,) into shape (41,)
# import torch
# from torch.utils.data import Dataset
# from Multi_change.preprocess_data import encode
# import json
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# #import cv2 as cv
# from imageio import imread
# from random import *
#
# class LEVIRCCDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#
#     def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 41, allow_unk = 0, max_iters=None):
#         """
#         :param data_folder: folder where image files are stored
#         :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param token_folder: folder where token files are stored
#         :param vocab_file: the name of vocab file
#         :param max_length: the maximum length of each caption sentence
#         :param max_iters: the maximum iteration when loading the data
#         :param allow_unk: whether to allow the tokens have unknow word or not
#         """
#         self.mean = [0.39073*255,  0.38623*255, 0.32989*255]
#         self.std = [0.15329*255,  0.14628*255, 0.13648*255]
#         self.list_path = list_path
#         self.split = split
#         self.max_length = max_length
#
#         assert self.split in {'train', 'val', 'test'}
#         self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
#         if vocab_file is not None:
#             with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
#                 self.word_vocab = json.load(f)
#             self.allow_unk = allow_unk
#         if not max_iters==None:
#             n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
#             self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
#         self.files = []
#         if split =='train':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name.split('-')[0])
#                 img_fileB = img_fileA.replace('A', 'B')
#
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#
#                 if '-' in name:
#                     token_id = name.split('-')[-1]
#                 else:
#                     token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name.split('-')[0]
#                 })
#         elif split =='val':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
#                 img_fileB = img_fileA.replace('A', 'B')
#
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#
#                 token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name
#                 })
#         elif split =='test':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
#                 img_fileB = img_fileA.replace('A', 'B')
#
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#
#                 token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name
#                 })
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         datafiles = self.files[index]
#         name = datafiles["name"]
#
#         imgA = datafiles["imgA"]
#         imgB = datafiles["imgB"]
#         seg_label = datafiles["seg_label"]
#
#         imgA = np.asarray(imgA, np.float32)
#         imgB = np.asarray(imgB, np.float32)
#         imgA = imgA.transpose(2, 0, 1)
#         imgB = imgB.transpose(2, 0, 1)
#         seg_label = seg_label.transpose(2, 0, 1)[0]
#         seg_label[seg_label==255] = 2
#         seg_label[seg_label==128] = 1
#
#
#         for i in range(len(self.mean)):
#             imgA[i,:,:] -= self.mean[i]
#             imgA[i,:,:] /= self.std[i]
#             imgB[i,:,:] -= self.mean[i]
#             imgB[i,:,:] /= self.std[i]
#         if datafiles["token"] is not None:
#             caption = open(datafiles["token"])
#             caption = caption.read()
#             caption_list = json.loads(caption)
#
#             #token = np.zeros((1, self.max_length), dtype=int)
#             #j = randint(0, len(caption_list) - 1)
#             #tokens_encode = encode(caption_list[j], self.word_vocab,
#             #            allow_unk=self.allow_unk == 1)
#             #token[0, :len(tokens_encode)] = tokens_encode
#             #token_len = len(tokens_encode)
#
#             token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
#             token_all_len = np.zeros((len(caption_list),1),dtype=int)
#             for j, tokens in enumerate(caption_list):
#                 nochange_cap = ['<START>', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '<END>']
#                 if self.split == 'train' and nochange_cap in caption_list:
#                     tokens = nochange_cap
#                 tokens_encode = encode(tokens, self.word_vocab,
#                                     allow_unk=self.allow_unk == 1)
#                 token_all[j,:len(tokens_encode)] = tokens_encode
#                 token_all_len[j] = len(tokens_encode)
#             if datafiles["token_id"] is not None:
#                 id = int(datafiles["token_id"])
#                 token = token_all[id]
#                 token_len = token_all_len[id].item()
#             else:
#                 j = randint(0, len(caption_list) - 1)
#                 token = token_all[j]
#                 token_len = token_all_len[j].item()
#         else:
#             token_all = np.zeros(1, dtype=int)
#             token = np.zeros(1, dtype=int)
#             token_len = np.zeros(1, dtype=int)
#             token_all_len = np.zeros(1, dtype=int)
#
#         return imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name

####  Below code was provided as solution to the previous error ###################
# import torch
# from torch.utils.data import Dataset
# from Multi_change.preprocess_data import encode
# import json
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# #from cv2 import cv
# from imageio import imread
# from random import *
# import warnings
# # Suppress the PyTorch deprecation warning for attention masks
# warnings.filterwarnings("ignore", category=UserWarning, message="Support for mismatched key_padding_mask and attn_mask")
#
# class LEVIRCCDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#
#     def __init__(self, data_folder, list_path, split, token_folder = None, vocab_file = None, max_length = 41, allow_unk = 0, max_iters=None):
#         """
#         :param data_folder: folder where image files are stored
#         :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param token_folder: folder where token files are stored
#         :param vocab_file: the name of vocab file
#         :param max_length: the maximum length of each caption sentence
#         :param max_iters: the maximum iteration when loading the data
#         :param allow_unk: whether to allow the tokens have unknow word or not
#         """
#
#         self.mean = [0.39073*255, 0.38623*255, 0.32989*255]
#         self.std = [0.15329*255, 0.14628*255, 0.13648*255]
#         self.list_path = list_path
#         self.split = split
#         self.max_length = max_length
#         assert self.split in {'train', 'val', 'test'}
#         self.img_ids = [i_id.strip() for i_id in open(os.path.join(list_path + split + '.txt'))]
#         if vocab_file is not None:
#             with open(os.path.join(list_path + vocab_file + '.json'), 'r') as f:
#                 self.word_vocab = json.load(f)
#         self.allow_unk = allow_unk
#         if not max_iters==None:
#             n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
#             self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters-n_repeat*len(self.img_ids)]
#         self.files = []
#         if split =='train':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name.split('-')[0])
#                 img_fileB = img_fileA.replace('A', 'B')
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#                 if '-' in name:
#                     token_id = name.split('-')[-1]
#                 else:
#                     token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name.split('-')[0]
#                 })
#         elif split =='val':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
#                 img_fileB = img_fileA.replace('A', 'B')
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#                 token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name
#                 })
#         elif split =='test':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder + '/' + split +'/A/' + name)
#                 img_fileB = img_fileA.replace('A', 'B')
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#                 token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder + name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name
#                 })
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         datafiles = self.files[index]
#         name = datafiles["name"]
#         imgA = datafiles["imgA"]
#         imgB = datafiles["imgB"]
#         seg_label = datafiles["seg_label"]
#         imgA = np.asarray(imgA, np.float32)
#         imgB = np.asarray(imgB, np.float32)
#         imgA = imgA.transpose(2, 0, 1)
#         imgB = imgB.transpose(2, 0, 1)
#         seg_label = seg_label.transpose(2, 0, 1)[0]
#         seg_label[seg_label==255] = 2
#         seg_label[seg_label==128] = 1
#         for i in range(len(self.mean)):
#             imgA[i,:,:] -= self.mean[i]
#             imgA[i,:,:] /= self.std[i]
#             imgB[i,:,:] -= self.mean[i]
#             imgB[i,:,:] /= self.std[i]
#         if datafiles["token"] is not None:
#             caption = open(datafiles["token"])
#             caption = caption.read()
#             caption_list = json.loads(caption)
#             #token = np.zeros((1, self.max_length), dtype=int)
#             #j = randint(0, len(caption_list) - 1)
#             #tokens_encode = encode(caption_list[j], self.word_vocab,
#             # allow_unk=self.allow_unk == 1)
#             #token[0, :len(tokens_encode)] = tokens_encode
#             #token_len = len(tokens_encode)
#             token_all = np.zeros((len(caption_list),self.max_length),dtype=int)
#             token_all_len = np.zeros((len(caption_list),1),dtype=int)
#             for j, tokens in enumerate(caption_list):
#                 nochange_cap = ['', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '']
#                 # Fixed: Check if any caption indicates "no change" (case-insensitive)
#                 is_no_change = any('no change' in str(caption).lower() or 'unchanged' in str(caption).lower() for caption in caption_list)
#                 if self.split == 'train' and is_no_change:
#                     tokens = nochange_cap  # Use fixed no-change caption
#                 tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
#                 # FIXED: Truncate tokens_encode to max_length to prevent broadcasting error
#                 if len(tokens_encode) > self.max_length:
#                     tokens_encode = tokens_encode[:self.max_length]
#                 # Ensure at least SOS if empty, and add EOS if not present (adjust if encode already adds them)
#                 if len(tokens_encode) == 0:
#                     sos_id = self.word_vocab.get('<sos>', 1)  # Default SOS ID if not found
#                     eos_id = self.word_vocab.get('<eos>', 2)  # Default EOS ID
#                     tokens_encode = [sos_id]
#                 if tokens_encode[-1] != self.word_vocab.get('<eos>', 2):  # Add EOS if missing
#                     eos_id = self.word_vocab.get('<eos>', 2)
#                     if len(tokens_encode) < self.max_length:
#                         tokens_encode.append(eos_id)
#                     else:
#                         tokens_encode[-1] = eos_id  # Replace last token with EOS if at max
#                 token_all[j, :len(tokens_encode)] = tokens_encode
#                 token_all_len[j] = len(tokens_encode)
#             if datafiles["token_id"] is not None:
#                 id = int(datafiles["token_id"])
#                 token = token_all[id]
#                 token_len = token_all_len[id].item()
#             else:
#                 j = randint(0, len(caption_list) - 1)
#                 token = token_all[j]
#                 token_len = token_all_len[j].item()
#         else:
#             token_all = np.zeros(1, dtype=int)
#             token = np.zeros(1, dtype=int)
#             token_len = np.zeros(1, dtype=int)
#             token_all_len = np.zeros(1, dtype=int)
#         return imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(token_len), name


############### New file to include previous truncations and fic the file path naming #####
#
# import torch
# from torch.utils.data import Dataset
# from preprocess_data import encode
# import json
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# # from cv2 import cv
# from imageio import imread
# from random import *
# import warnings
#
# # Suppress the PyTorch deprecation warning for attention masks
# warnings.filterwarnings("ignore", category=UserWarning, message="Support for mismatched key_padding_mask and attn_mask")
#
#
# class LEVIRCCDataset(Dataset):
#     """
#     A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
#     """
#
#     def __init__(self, data_folder, list_path, split, token_folder=None, vocab_file=None, max_length=41, allow_unk=0,
#                  max_iters=None):
#         """
#         :param data_folder: folder where image files are stored
#         :param list_path: folder where the file name-lists of Train/val/test.txt sets are stored
#         :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
#         :param token_folder: folder where token files are stored
#         :param vocab_file: the name of vocab file
#         :param max_length: the maximum length of each caption sentence
#         :param max_iters: the maximum iteration when loading the data
#         :param allow_unk: whether to allow the tokens have unknow word or not
#         """
#
#         self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
#         self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
#         self.list_path = list_path
#         self.split = split.lower()  # Normalize to lowercase for consistency
#         self.max_length = max_length
#         assert self.split in {'train', 'val', 'test'}
#
#         # FIXED: Proper path joining for the split file (prevents concatenation bug)
#         split_file = os.path.join(list_path, self.split + '.txt')
#         if not os.path.exists(split_file):
#             raise FileNotFoundError(
#                 f"Split file not found: {split_file}. Ensure {self.split}.txt exists in {list_path}.")
#
#         self.img_ids = [i_id.strip() for i_id in open(split_file)]
#
#         if vocab_file is not None:
#             vocab_path = os.path.join(list_path, vocab_file + '.json')
#             with open(vocab_path, 'r') as f:
#                 self.word_vocab = json.load(f)
#         self.allow_unk = allow_unk
#         if not max_iters == None:
#             n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
#             self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]
#         self.files = []
#         if self.split == 'train':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder, self.split, 'A', name.split('-')[0])
#                 img_fileB = img_fileA.replace('A', 'B')
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#                 if '-' in name:
#                     token_id = name.split('-')[-1]
#                 else:
#                     token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder, name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name.split('-')[0]
#                 })
#         elif self.split == 'val':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder, self.split, 'A', name)
#                 img_fileB = img_fileA.replace('A', 'B')
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#                 token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder, name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name
#                 })
#         elif self.split == 'test':
#             for name in self.img_ids:
#                 img_fileA = os.path.join(data_folder, self.split, 'A', name)
#                 img_fileB = img_fileA.replace('A', 'B')
#                 imgA = imread(img_fileA)
#                 imgB = imread(img_fileB)
#                 seg_label = imread(img_fileA.replace('A', 'label'))
#                 token_id = None
#                 if token_folder is not None:
#                     token_file = os.path.join(token_folder, name.split('.')[0] + '.txt')
#                 else:
#                     token_file = None
#                 self.files.append({
#                     "imgA": imgA,
#                     "imgB": imgB,
#                     "seg_label": seg_label,
#                     "token": token_file,
#                     "token_id": token_id,
#                     "name": name
#                 })
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, index):
#         datafiles = self.files[index]
#         name = datafiles["name"]
#         imgA = datafiles["imgA"]
#         imgB = datafiles["imgB"]
#         seg_label = datafiles["seg_label"]
#         imgA = np.asarray(imgA, np.float32)
#         imgB = np.asarray(imgB, np.float32)
#         imgA = imgA.transpose(2, 0, 1)
#         imgB = imgB.transpose(2, 0, 1)
#         seg_label = seg_label.transpose(2, 0, 1)[0]
#         seg_label[seg_label == 255] = 2
#         seg_label[seg_label == 128] = 1
#         for i in range(len(self.mean)):
#             imgA[i, :, :] -= self.mean[i]
#             imgA[i, :, :] /= self.std[i]
#             imgB[i, :, :] -= self.mean[i]
#             imgB[i, :, :] /= self.std[i]
#         if datafiles["token"] is not None and os.path.exists(datafiles["token"]):
#             with open(datafiles["token"], 'r') as caption_file:
#                 caption = caption_file.read()
#             caption_list = json.loads(caption)
#             token_all = np.zeros((len(caption_list), self.max_length), dtype=int)
#             token_all_len = np.zeros((len(caption_list), 1), dtype=int)
#             for j, tokens in enumerate(caption_list):
#                 # # the below is commented  that removes the override, uses the saved tokens (which already include /), and enforces a safe truncation that preserves
#                 # nochange_cap = ['', 'the', 'scene', 'is', 'the', 'same', 'as', 'before', '']
#                 # # Fixed: Check if any caption indicates "no change" (case-insensitive)
#                 # is_no_change = any(
#                 #     'no change' in str(caption).lower() or 'unchanged' in str(caption).lower() for caption in
#                 #     caption_list)
#                 # if self.split == 'train' and is_no_change:
#                 #     tokens = nochange_cap  # Use fixed no-change caption
#                 tokens_encode = encode(tokens, self.word_vocab, allow_unk=self.allow_unk == 1)
#                 # FIXED: Truncate tokens_encode to max_length to prevent broadcasting error
#                 if len(tokens_encode) > self.max_length:
#                     tokens_encode = tokens_encode[:self.max_length]
#                 # Ensure at least SOS if empty, and add EOS if not present (adjust if encode already adds them)
#                 if len(tokens_encode) == 0:
#                     sos_id = self.word_vocab.get('<sos>', 1)  # Default SOS ID if not found
#                     eos_id = self.word_vocab.get('<eos>', 2)  # Default EOS ID
#                     tokens_encode = [sos_id]
#                 if tokens_encode[-1] != self.word_vocab.get('<eos>', 2):  # Add EOS if missing
#                     eos_id = self.word_vocab.get('<eos>', 2)
#                     if len(tokens_encode) < self.max_length:
#                         tokens_encode.append(eos_id)
#                     else:
#                         tokens_encode[-1] = eos_id  # Replace last token with EOS if at max
#                 token_all[j, :len(tokens_encode)] = tokens_encode
#                 token_all_len[j] = len(tokens_encode)
#             if datafiles["token_id"] is not None:
#                 id = int(datafiles["token_id"])
#                 token = token_all[id]
#                 token_len = token_all_len[id].item()
#             else:
#                 j = randint(0, len(caption_list) - 1)
#                 token = token_all[j]
#                 token_len = token_all_len[j].item()
#         else:
#             token_all = np.zeros(1, dtype=int)
#             token = np.zeros(1, dtype=int)
#             token_len = np.zeros(1, dtype=int)
#             token_all_len = np.zeros(1, dtype=int)
#         return imgA.copy(), imgB.copy(), seg_label.copy(), token_all.copy(), token_all_len.copy(), token.copy(), np.array(
#             token_len), name


########################################################################################################################

# data/LEVIR_MCI.py
import os
import sys
import json
import numpy as np
from random import randint
from torch.utils.data import Dataset
from imageio import imread

# Robust import of encode from preprocess_data.py at repo root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
try:
    from preprocess_data import encode  # SPECIAL_TOKENS: <NULL>, <UNK>, <START>, <END>
except Exception:
    # Minimal fallback so runtime never breaks if pathing is odd
    def encode(seq_tokens, token_to_idx, allow_unk=False):
        out = []
        for t in seq_tokens:
            if t in token_to_idx:
                out.append(token_to_idx[t])
            elif allow_unk:
                out.append(token_to_idx['<UNK>'])
            else:
                raise KeyError(f'Token "{t}" not in vocab')
        return out


class LEVIRCCDataset(Dataset):
    """
    LEVIR-MCI dataset loader.
    - Uses the 5 ground-truth captions as-is (no overrides).
    - Safe truncation to max_length; forces last token to <END>.
    - Returns both a single training caption (token, token_len) and the full 5-reference set (token_all, token_all_len).
    """

    def __init__(self, data_folder, list_path, split,
                 token_folder=None, vocab_file=None,
                 max_length=46, allow_unk=1, max_iters=None):
        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
        self.split = split.lower()
        self.data_folder = data_folder
        self.list_path = list_path
        self.token_folder = token_folder
        self.max_length = int(max_length)
        self.allow_unk = int(allow_unk)

        assert self.split in {'train', 'val', 'test'}

        split_file = os.path.join(self.list_path, f'{self.split}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f'Missing split file: {split_file}')

        self.img_ids = [line.strip() for line in open(split_file, 'r')]
        if max_iters is not None:
            n_repeat = int(np.ceil(max_iters / len(self.img_ids)))
            self.img_ids = self.img_ids * n_repeat + self.img_ids[:max_iters - n_repeat * len(self.img_ids)]

        # load vocab
        if vocab_file is not None:
            vocab_path = os.path.join(self.list_path, vocab_file + '.json')
            if not os.path.exists(vocab_path):
                raise FileNotFoundError(f'Missing vocab json: {vocab_path}')
            with open(vocab_path, 'r') as f:
                self.word_vocab = json.load(f)
        else:
            self.word_vocab = None

        # Build file records (store paths; read images in __getitem__)
        self.files = []
        if self.split == 'train':
            for name in self.img_ids:
                base = name.split('-')[0]  # support train_XXXX-<token_id>
                imgA_path = os.path.join(self.data_folder, self.split, 'A', base)
                imgB_path = imgA_path.replace('A', 'B')
                seg_path = imgA_path.replace('A', 'label')
                token_id = name.split('-')[-1] if '-' in name else None
                tok_path = os.path.join(self.token_folder, base.split('.')[0] + '.txt') if self.token_folder else None
                self.files.append({
                    'imgA': imgA_path, 'imgB': imgB_path, 'seg': seg_path,
                    'token': tok_path, 'token_id': token_id, 'name': base
                })
        else:
            for name in self.img_ids:
                imgA_path = os.path.join(self.data_folder, self.split, 'A', name)
                imgB_path = imgA_path.replace('A', 'B')
                seg_path = imgA_path.replace('A', 'label')
                tok_path = os.path.join(self.token_folder, name.split('.')[0] + '.txt') if self.token_folder else None
                self.files.append({
                    'imgA': imgA_path, 'imgB': imgB_path, 'seg': seg_path,
                    'token': tok_path, 'token_id': None, 'name': name
                })

    def __len__(self):
        return len(self.files)

    def _read_rgb(self, path):
        arr = imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr.astype(np.float32)

    def _read_seg(self, path):
        seg = imread(path)
        if seg.ndim == 2:
            seg = np.expand_dims(seg, axis=-1)
        seg = seg.transpose(2, 0, 1)[0]  # first channel
        seg = seg.astype(np.int64)
        seg[seg == 255] = 2  # building
        seg[seg == 128] = 1  # road
        return seg

    def _norm(self, img):
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        for c in range(3):
            img[c] = (img[c] - self.mean[c]) / self.std[c]
        return img

    def _encode_all_captions(self, token_file):
        if token_file is None or (not os.path.exists(token_file)) or (self.word_vocab is None):
            return (np.zeros(1, dtype=np.int64),
                    np.zeros(1, dtype=np.int64))

        with open(token_file, 'r') as f:
            caps = json.load(f)  # list of 5 token lists, each already with <START>...<END>

        num_caps = len(caps)
        token_all = np.zeros((num_caps, self.max_length), dtype=np.int64)
        token_all_len = np.zeros((num_caps, 1), dtype=np.int64)

        END = self.word_vocab['<END>']

        for j, tokens in enumerate(caps):
            ids = encode(tokens, self.word_vocab, allow_unk=(self.allow_unk == 1))

            # safe truncation to max_length; force <END> at last position
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
                ids[-1] = END
            else:
                if len(ids) == 0:
                    ids = [END]
                if ids[-1] != END:
                    if len(ids) < self.max_length:
                        ids = ids + [END]
                    else:
                        ids[-1] = END

            token_all[j, :len(ids)] = np.array(ids, dtype=np.int64)
            token_all_len[j, 0] = len(ids)

        return token_all, token_all_len

    def __getitem__(self, idx):
        rec = self.files[idx]
        imgA = self._norm(self._read_rgb(rec['imgA']))
        imgB = self._norm(self._read_rgb(rec['imgB']))
        seg = self._read_seg(rec['seg'])

        token_all, token_all_len = self._encode_all_captions(rec['token'])

        # choose one caption for training; val/test keep token_all for references
        if isinstance(token_all, np.ndarray) and token_all.ndim == 2 and token_all.shape[0] > 1:
            if rec['token_id'] is not None:
                j = int(rec['token_id'])
            else:
                j = randint(0, token_all.shape[0] - 1)
            token = token_all[j]
            token_len = int(token_all_len[j, 0])
        else:
            token = np.zeros(1, dtype=np.int64)
            token_len = 0

        return (
            imgA.copy(), imgB.copy(), seg.copy(),
            token_all.copy(), token_all_len.copy(),
            token.copy(), np.array(token_len), rec['name']
        )