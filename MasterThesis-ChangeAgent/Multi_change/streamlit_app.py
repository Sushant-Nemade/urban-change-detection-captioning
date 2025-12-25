import os
import io
import json
import time
import numpy as np
import torch
import cv2
import streamlit as st
from PIL import Image

# repo-local imports (run from repo root)

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer

# Fixed config (hidden from the UI)

BACKBONE = 'segformer-mit_b1'
TARGET_SIZE = 256       # model input size (hidden)
DEFAULT_BEAM = 5        # caption beam size (hidden)
LIST_PATH_DEFAULT = './data/LEVIR_MCI'
AUG_VOCAB_NAME = 'vocab'
ORIG_VOCAB_NAME = 'vocab_original'

# Normalization used in your dataset/model

MEAN = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
STD  = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

def to_rgb3(pil_img: Image.Image) -> np.ndarray:
    # Force RGB (handles L/LA/RGBA)
    return np.array(pil_img.convert("RGB"), dtype=np.float32)

def resize_hwc(img_hwc: np.ndarray, size: int) -> np.ndarray:
    if img_hwc.shape[0] == size and img_hwc.shape[1] == size:
        return img_hwc
    return cv2.resize(img_hwc, (size, size), interpolation=cv2.INTER_LINEAR)

def norm_to_chw(img_hwc: np.ndarray) -> np.ndarray:
    chw = np.transpose(img_hwc, (2, 0, 1))
    for c in range(3):
        chw[c] = (chw[c] - MEAN[c]) / STD[c]
    return chw

def select_vocab_for_ckpt(dec_sd, list_path):
    # Infer vocab size and max_len from decoder buffers
    ck_vocab_size = int(dec_sd['wdc.weight'].shape[0])
    ck_max_len    = int(dec_sd['position_encoding.pe'].shape[0])

    # Decide which vocab file to use
    vocab_name = AUG_VOCAB_NAME if ck_vocab_size != 468 else ORIG_VOCAB_NAME
    vocab_path = os.path.join(list_path, vocab_name + '.json')

    caption_enabled = True
    if not os.path.exists(vocab_path):
        caption_enabled = False
        word_vocab = {'<NULL>':0,'<UNK>':1,'<START>':2,'<END>':3}
    else:
        with open(vocab_path, 'r') as f:
            word_vocab = json.load(f)
        if len(word_vocab) != ck_vocab_size:
            raise RuntimeError(f'Vocab size mismatch: {vocab_name} has {len(word_vocab)} vs ckpt {ck_vocab_size}')
    return word_vocab, vocab_name, ck_vocab_size, ck_max_len, caption_enabled

@st.cache_resource(show_spinner=False)
def load_model(checkpoint_path: str, list_path: str, device_str: str):
    device = torch.device(device_str)
    ckpt = torch.load(checkpoint_path, map_location=device)
    dec_sd = ckpt['decoder_dict']

    word_vocab, vocab_used, ck_vocab_size, ck_max_len, caption_enabled = select_vocab_for_ckpt(dec_sd, list_path)
    idx_to_word = {v: k for k, v in word_vocab.items()}
    specials = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}

    # Build model graph (fixed backbone segformer-mit_b1)
    encoder = Encoder(BACKBONE).eval().to(device)
    encoder_trans = AttentiveEncoder(
        train_stage=None, n_layers=3,
        feature_size=[16, 16, 512], heads=8, dropout=0.1
    ).eval().to(device)
    decoder = DecoderTransformer(
        encoder_dim=512, feature_dim=512,
        vocab_size=len(word_vocab), max_lengths=ck_max_len,
        word_vocab=word_vocab, n_head=8,
        n_layers=1, dropout=0.1
    ).eval().to(device)

    # Load weights
    encoder.load_state_dict(ckpt['encoder_dict'], strict=True)
    encoder_trans.load_state_dict(ckpt['encoder_trans_dict'], strict=False)
    decoder.load_state_dict(dec_sd, strict=True)

    return {
        'device': device,
        'encoder': encoder,
        'encoder_trans': encoder_trans,
        'decoder': decoder,
        'word_vocab': word_vocab,
        'idx_to_word': idx_to_word,
        'specials': specials,
        'ck_max_len': ck_max_len,
        'caption_enabled': caption_enabled,
        'vocab_used': vocab_used
    }

def prepare_images(pil_A: Image.Image, pil_B: Image.Image, device=None):
    npA = to_rgb3(pil_A)
    npB = to_rgb3(pil_B)
    origA = (npA.shape[0], npA.shape[1])
    origB = (npB.shape[0], npB.shape[1])

    npA = resize_hwc(npA, TARGET_SIZE)
    npB = resize_hwc(npB, TARGET_SIZE)
    chwA = norm_to_chw(npA)
    chwB = norm_to_chw(npB)

    tenA = torch.from_numpy(chwA).float().unsqueeze(0).to(device, non_blocking=True)
    tenB = torch.from_numpy(chwB).float().unsqueeze(0).to(device, non_blocking=True)
    return tenA, tenB, origA, origB

@torch.no_grad()
def forward_once(encoder, encoder_trans, imgA, imgB):
    feat1_list, feat2_list = encoder(imgA, imgB)
    feat1_cap, feat2_cap, seg_pre = encoder_trans(feat1_list, feat2_list)
    return feat1_cap, feat2_cap, seg_pre

@torch.no_grad()
def infer_caption(decoder, feat1, feat2, idx_to_word, specials):
    if decoder is None:
        return ''
    # Hidden default beam size = 5 for better fluency
    seq = decoder.sample_beam(feat1, feat2, k=DEFAULT_BEAM)
    if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
        seq = seq[0]
    tokens = [idx_to_word.get(i, '<UNK>') for i in seq if i not in specials]
    return ' '.join(tokens).strip()

@torch.no_grad()
def infer_mask(seg_pre, save_size=None):
    pred = np.argmax(seg_pre.detach().cpu().numpy(), axis=1)[0].astype(np.uint8)
    if save_size is not None and (pred.shape[0], pred.shape[1]) != save_size:
        pred = cv2.resize(pred, (save_size[1], save_size[0]), interpolation=cv2.INTER_NEAREST)

    # road=1 -> yellow (RGB), building=2 -> red (RGB)
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[pred == 1] = (255, 255, 0)
    rgb[pred == 2] = (255, 0, 0)
    return pred, rgb

def save_results(out_dir, base_name, pilA, pilB, mask_rgb, caption):
    os.makedirs(out_dir, exist_ok=True)
    path_A = os.path.join(out_dir, f'{base_name}_A.png')
    path_B = os.path.join(out_dir, f'{base_name}_B.png')
    pilA.save(path_A)
    pilB.save(path_B)
    path_mask = os.path.join(out_dir, f'{base_name}_mask.png')
    cv2.imwrite(path_mask, mask_rgb[:, :, ::-1])  # RGB->BGR
    path_cap = os.path.join(out_dir, f'{base_name}_caption.txt')
    with open(path_cap, 'w', encoding='utf-8') as f:
        f.write(caption if caption else '')
    return {'A': path_A, 'B': path_B, 'mask': path_mask, 'caption': path_cap}

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Change Agent — Inference", layout="wide")

st.title("Change Agent — Bi-temporal Change Interpretation")
st.caption("Upload Image A and Image B, then click Predict to get a change mask and caption.")

with st.sidebar:
    st.header("Model & Output")
    checkpoint_path = st.text_input("Checkpoint (.pth)", value="", help="Path to your trained checkpoint (.pth)")
    save_mask_size = st.selectbox("Save mask at size of", ["target (256x256)", "A original", "B original"], index=1)
    out_dir = st.text_input("Output directory", value="./app_outputs", help="Where Save Results will be written")
    load_btn = st.button("Load/Reload Model", type="primary", use_container_width=True)

# File uploaders

colA, colB = st.columns(2)
with colA:
    fileA = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg", "tif", "tiff"])
with colB:
    fileB = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg", "tif", "tiff"])

# Session storage

if "model" not in st.session_state:
    st.session_state.model = None
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

def can_load():
    return os.path.isfile(checkpoint_path)

# Load model

if load_btn:
    try:
        if not can_load():
            st.error("Please provide a valid checkpoint path.")
        else:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
            st.session_state.model = load_model(
                checkpoint_path=checkpoint_path,
                list_path=LIST_PATH_DEFAULT,
                device_str=device_str
            )
            m = st.session_state.model
            st.success(f"Model loaded (device: {device_str}). Vocab: {m['vocab_used']}. "
                       f"Caption: {'ON' if m['caption_enabled'] else 'OFF'}. Backbone: {BACKBONE}")
    except Exception as e:
        st.exception(e)

# Predict / Save / Clear controls

if fileA is not None and fileB is not None:
    imgA_pil = Image.open(io.BytesIO(fileA.read()))
    imgB_pil = Image.open(io.BytesIO(fileB.read()))
    with colA:
        st.image(imgA_pil, caption="Image A", use_container_width=True)
    with colB:
        st.image(imgB_pil, caption="Image B", use_container_width=True)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        pred_btn = st.button("Predict", type="primary")
    with c2:
        save_btn = st.button("Save Results")
    with c3:
        clear_btn = st.button("Clear")

    if pred_btn:
        if st.session_state.model is None:
            st.error("Please load the model first.")
        else:
            try:
                m = st.session_state.model
                tenA, tenB, origA, origB = prepare_images(imgA_pil, imgB_pil, device=m['device'])
                f1, f2, seg_pre = forward_once(m['encoder'], m['encoder_trans'], tenA, tenB)

                # mask sizing
                save_hw = None
                if save_mask_size.startswith("A"):
                    save_hw = origA
                elif save_mask_size.startswith("B"):
                    save_hw = origB
                pred_lbl, pred_rgb = infer_mask(seg_pre, save_size=save_hw)

                caption = ""
                if m['caption_enabled']:
                    caption = infer_caption(m['decoder'], f1, f2, m['idx_to_word'], m['specials'])

                st.subheader("Results")
                st.image(pred_rgb, caption="Predicted Mask (road=yellow, building=red)", use_container_width=True)
                st.write(f"Caption: {caption if caption else '(caption disabled or unavailable)'}")

                st.session_state.last_pred = {
                    'imgA_pil': imgA_pil,
                    'imgB_pil': imgB_pil,
                    'mask_rgb': pred_rgb,
                    'caption': caption
                }

            except Exception as e:
                st.exception(e)

    if save_btn:
        if st.session_state.last_pred is None:
            st.warning("Nothing to save yet. Please run Predict first.")
        else:
            try:
                base = time.strftime("%Y%m%d_%H%M%S")
                paths = save_results(
                    out_dir, base,
                    st.session_state.last_pred['imgA_pil'],
                    st.session_state.last_pred['imgB_pil'],
                    st.session_state.last_pred['mask_rgb'],
                    st.session_state.last_pred['caption']
                )
                st.success(f"Saved:\nA: {paths['A']}\nB: {paths['B']}\nMask: {paths['mask']}\nCaption: {paths['caption']}")
            except Exception as e:
                st.exception(e)

    if clear_btn:
        st.session_state.last_pred = None
        st.rerun()
else:
    st.info("Upload both Image A and Image B to proceed.")



################## The below streamlit code works but it is more detialed in terms of models and setting which may not be useful for every user ##############
# import os
# import io
# import json
# import time
# import numpy as np
# import torch
# import cv2
# import streamlit as st
# from PIL import Image
#
# # repo-local imports (run from repo root)
#
# from model.model_encoder_att import Encoder, AttentiveEncoder
# from model.model_decoder import DecoderTransformer
#
# # Normalization used in your dataset/model
#
# MEAN = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
# STD  = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]
#
# def to_rgb3(pil_img: Image.Image) -> np.ndarray:
#     # Force RGB (handles L/LA/RGBA)
#     return np.array(pil_img.convert("RGB"), dtype=np.float32)
#
# def resize_hwc(img_hwc: np.ndarray, size: int) -> np.ndarray:
#     if img_hwc.shape[0] == size and img_hwc.shape[1] == size:
#         return img_hwc
#     return cv2.resize(img_hwc, (size, size), interpolation=cv2.INTER_LINEAR)
#
# def norm_to_chw(img_hwc: np.ndarray) -> np.ndarray:
#     chw = np.transpose(img_hwc, (2, 0, 1))
#     for c in range(3):
#         chw[c] = (chw[c] - MEAN[c]) / STD[c]
#     return chw
#
# def select_vocab_for_ckpt(dec_sd, list_path, aug_name, orig_name):
#     ck_vocab_size = int(dec_sd['wdc.weight'].shape[0])
#     ck_max_len    = int(dec_sd['position_encoding.pe'].shape[0])
#
#     # Decide which vocab to load
#     vocab_name = aug_name
#     if ck_vocab_size == 468:
#         vocab_name = orig_name  # original MCI
#     vocab_path = os.path.join(list_path, vocab_name + '.json')
#
#     caption_enabled = True
#     if not os.path.exists(vocab_path):
#         caption_enabled = False
#         word_vocab = {'<NULL>':0,'<UNK>':1,'<START>':2,'<END>':3}
#     else:
#         with open(vocab_path, 'r') as f:
#             word_vocab = json.load(f)
#         if len(word_vocab) != ck_vocab_size:
#             raise RuntimeError(f'Vocab size mismatch: {vocab_name} has {len(word_vocab)} vs ckpt {ck_vocab_size}')
#     return word_vocab, vocab_name, ck_vocab_size, ck_max_len, caption_enabled
#
# @st.cache_resource(show_spinner=False)
# def load_model(checkpoint_path: str,
#                list_path: str,
#                network: str,
#                encoder_dim: int,
#                feat_size: int,
#                n_heads: int,
#                n_layers: int,
#                decoder_n_layers: int,
#                feature_dim: int,
#                dropout: float,
#                vocab_file: str,
#                orig_vocab_file: str,
#                device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
#     device = torch.device(device_str)
#     ckpt = torch.load(checkpoint_path, map_location=device)
#     dec_sd = ckpt['decoder_dict']
#
#     word_vocab, vocab_used, ck_vocab_size, ck_max_len, caption_enabled = select_vocab_for_ckpt(
#         dec_sd, list_path, vocab_file, orig_vocab_file
#     )
#     idx_to_word = {v: k for k, v in word_vocab.items()}
#     specials = {word_vocab['<START>'], word_vocab['<END>'], word_vocab['<NULL>']}
#
#     # Build model graph
#     encoder = Encoder(network).eval().to(device)
#     encoder_trans = AttentiveEncoder(
#         train_stage=None, n_layers=n_layers,
#         feature_size=[feat_size, feat_size, encoder_dim],
#         heads=n_heads, dropout=dropout
#     ).eval().to(device)
#     decoder = DecoderTransformer(
#         encoder_dim=encoder_dim, feature_dim=feature_dim,
#         vocab_size=len(word_vocab), max_lengths=ck_max_len,
#         word_vocab=word_vocab, n_head=n_heads,
#         n_layers=decoder_n_layers, dropout=dropout
#     ).eval().to(device)
#
#     # Load weights
#     encoder.load_state_dict(ckpt['encoder_dict'], strict=True)
#     encoder_trans.load_state_dict(ckpt['encoder_trans_dict'], strict=False)
#     decoder.load_state_dict(dec_sd, strict=True)
#
#     return {
#         'device': device,
#         'encoder': encoder,
#         'encoder_trans': encoder_trans,
#         'decoder': decoder,
#         'word_vocab': word_vocab,
#         'idx_to_word': idx_to_word,
#         'specials': specials,
#         'ck_max_len': ck_max_len,
#         'caption_enabled': caption_enabled,
#         'vocab_used': vocab_used
#     }
#
# def prepare_images(pil_A: Image.Image, pil_B: Image.Image, target_size: int = 256, device=None):
#     npA = to_rgb3(pil_A)  # HWC float32
#     npB = to_rgb3(pil_B)
#     origA = (npA.shape[0], npA.shape[1])
#     origB = (npB.shape[0], npB.shape[1])
#
#     npA = resize_hwc(npA, target_size)
#     npB = resize_hwc(npB, target_size)
#     chwA = norm_to_chw(npA)
#     chwB = norm_to_chw(npB)
#
#     tenA = torch.from_numpy(chwA).float().unsqueeze(0).to(device, non_blocking=True)
#     tenB = torch.from_numpy(chwB).float().unsqueeze(0).to(device, non_blocking=True)
#     return tenA, tenB, origA, origB
#
# @torch.no_grad()
# def forward_once(encoder, encoder_trans, imgA, imgB):
#     feat1_list, feat2_list = encoder(imgA, imgB)
#     feat1_cap, feat2_cap, seg_pre = encoder_trans(feat1_list, feat2_list)
#     return feat1_cap, feat2_cap, seg_pre
#
# @torch.no_grad()
# def infer_caption(decoder, feat1, feat2, word_vocab, idx_to_word, specials, beam_size=5):
#     if decoder is None:
#         return ''
#     if beam_size > 1:
#         seq = decoder.sample_beam(feat1, feat2, k=beam_size)
#     else:
#         seq = decoder.sample(feat1, feat2, k=1)
#     if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
#         seq = seq[0]
#     tokens = [idx_to_word.get(i, '<UNK>') for i in seq if i not in specials]
#     return ' '.join(tokens).strip()
#
# @torch.no_grad()
# def infer_mask(seg_pre, save_size=None):
#     # seg_pre: (1, C, H, W) logits
#     pred = np.argmax(seg_pre.detach().cpu().numpy(), axis=1)[0].astype(np.uint8)  # (H,W)
#     if save_size is not None and (pred.shape[0], pred.shape[1]) != save_size:
#         pred = cv2.resize(pred, (save_size[1], save_size[0]), interpolation=cv2.INTER_NEAREST)
#
#     # Colorize: road=1 -> yellow (RGB), building=2 -> red (RGB)
#     h, w = pred.shape
#     rgb = np.zeros((h, w, 3), dtype=np.uint8)
#     rgb[pred == 1] = (255, 255, 0)
#     rgb[pred == 2] = (255, 0, 0)
#     return pred, rgb
#
# def save_results(out_dir, base_name, pilA, pilB, mask_rgb, caption):
#     os.makedirs(out_dir, exist_ok=True)
#     # Save inputs
#     path_A = os.path.join(out_dir, f'{base_name}_A.png')
#     path_B = os.path.join(out_dir, f'{base_name}_B.png')
#     pilA.save(path_A)
#     pilB.save(path_B)
#     # Save mask (convert RGB->BGR for OpenCV write)
#     path_mask = os.path.join(out_dir, f'{base_name}_mask.png')
#     cv2.imwrite(path_mask, mask_rgb[:, :, ::-1])
#     # Save caption
#     path_cap = os.path.join(out_dir, f'{base_name}_caption.txt')
#     with open(path_cap, 'w', encoding='utf-8') as f:
#         f.write(caption if caption else '')
#     return {'A': path_A, 'B': path_B, 'mask': path_mask, 'caption': path_cap}
#
# # ------------------- Streamlit UI -------------------
#
# st.set_page_config(page_title="Change Agent — Inference", layout="wide")
#
# st.title("Change Agent — Bi-temporal Change Interpretation")
# st.caption("Upload Image A and Image B, then click Predict to get a change mask and caption.")
#
# with st.sidebar:
#     st.header("Model & Settings")
#
#     checkpoint_path = st.text_input(
#         "Checkpoint (.pth)",
#         value="",
#         help="Path to your trained checkpoint (.pth)"
#     )
#     list_path = st.text_input(
#         "List/Vocab folder",
#         value="./data/LEVIR_MCI",
#         help="Folder containing vocab.json and vocab_original.json"
#     )
#     vocab_file = st.text_input("Augmented vocab filename (no .json)", value="vocab")
#     orig_vocab_file = st.text_input("Original vocab filename (no .json)", value="vocab_original")
#
#     network = st.selectbox("Backbone", ["segformer-mit_b1", "segformer-mit_b0", "segformer-mit_b2"], index=0)
#     beam_size = st.number_input("Beam size (caption)", min_value=1, max_value=10, value=5, step=1)
#     target_size = st.number_input("Model input size", min_value=128, max_value=512, value=256, step=32)
#     save_mask_size = st.selectbox("Save mask at size of", ["target (256x256)", "A original", "B original"], index=1)
#     out_dir = st.text_input("Output directory", value="./app_outputs", help="Where Save Results will be written")
#     run_device = "cuda" if torch.cuda.is_available() else "cpu"
#     st.write(f"Device: {run_device}")
#
#     load_btn = st.button("Load/Reload Model", type="primary", use_container_width=True)
#
# # file uploaders
#
# colA, colB = st.columns(2)
# with colA:
#     fileA = st.file_uploader("Upload Image A", type=["png", "jpg", "jpeg", "tif", "tiff"])
# with colB:
#     fileB = st.file_uploader("Upload Image B", type=["png", "jpg", "jpeg", "tif", "tiff"])
#
# # Session storage
#
# if "model" not in st.session_state:
#     st.session_state.model = None
# if "last_pred" not in st.session_state:
#     st.session_state.last_pred = None
#
# # Load model (on demand)
#
# def can_load():
#     return os.path.isfile(checkpoint_path) and os.path.isdir(list_path) and len(checkpoint_path) > 0
#
# if load_btn:
#     try:
#         if not can_load():
#             st.error("Please provide a valid checkpoint path and list_path.")
#         else:
#             st.session_state.model = load_model(
#                 checkpoint_path=checkpoint_path,
#                 list_path=list_path,
#                 network=network,
#                 encoder_dim=512, feat_size=16,
#                 n_heads=8, n_layers=3,
#                 decoder_n_layers=1,
#                 feature_dim=512, dropout=0.1,
#                 vocab_file=vocab_file,
#                 orig_vocab_file=orig_vocab_file,
#                 device_str=run_device
#             )
#             st.success(f"Model loaded. Vocab: {st.session_state.model['vocab_used']}. "
#                        f"Caption: {'ON' if st.session_state.model['caption_enabled'] else 'OFF'}.")
#     except Exception as e:
#         st.exception(e)
#
# # Preview and Predict
#
# if fileA is not None and fileB is not None:
#     # Show inputs
#     imgA_pil = Image.open(io.BytesIO(fileA.read()))
#     imgB_pil = Image.open(io.BytesIO(fileB.read()))
#     with colA:
#         st.image(imgA_pil, caption="Image A", use_container_width=True)
#     with colB:
#         st.image(imgB_pil, caption="Image B", use_container_width=True)
#
#     # Buttons row
#     c1, c2, c3 = st.columns([1,1,1])
#     with c1:
#         pred_btn = st.button("Predict", type="primary")
#     with c2:
#         save_btn = st.button("Save Results")
#     with c3:
#         clear_btn = st.button("Clear")
#
#     if pred_btn:
#         if st.session_state.model is None:
#             st.error("Please load the model first.")
#         else:
#             try:
#                 m = st.session_state.model
#                 tenA, tenB, origA, origB = prepare_images(
#                     imgA_pil, imgB_pil, target_size=int(target_size), device=m['device']
#                 )
#                 f1, f2, seg_pre = forward_once(m['encoder'], m['encoder_trans'], tenA, tenB)
#
#                 # mask sizing
#                 save_hw = None
#                 if save_mask_size.startswith("A"):
#                     save_hw = origA
#                 elif save_mask_size.startswith("B"):
#                     save_hw = origB
#                 pred_lbl, pred_rgb = infer_mask(seg_pre, save_size=save_hw)
#
#                 caption = ""
#                 if m['caption_enabled']:
#                     caption = infer_caption(
#                         m['decoder'], f1, f2, m['word_vocab'], m['idx_to_word'], m['specials'], beam_size=int(beam_size)
#                     )
#
#                 # show results
#                 st.subheader("Results")
#                 st.image(pred_rgb, caption="Predicted Mask (road=yellow, building=red)", use_container_width=True)
#                 st.write(f"Caption: {caption if caption else '(caption disabled or unavailable)'}")
#
#                 # store last pred for saving
#                 st.session_state.last_pred = {
#                     'imgA_pil': imgA_pil,
#                     'imgB_pil': imgB_pil,
#                     'mask_rgb': pred_rgb,
#                     'caption': caption
#                 }
#
#             except Exception as e:
#                 st.exception(e)
#
#     if save_btn:
#         if st.session_state.last_pred is None:
#             st.warning("Nothing to save yet. Please run Predict first.")
#         else:
#             try:
#                 base = time.strftime("%Y%m%d_%H%M%S")
#                 paths = save_results(
#                     out_dir, base,
#                     st.session_state.last_pred['imgA_pil'],
#                     st.session_state.last_pred['imgB_pil'],
#                     st.session_state.last_pred['mask_rgb'],
#                     st.session_state.last_pred['caption']
#                 )
#                 st.success(f"Saved:\nA: {paths['A']}\nB: {paths['B']}\nMask: {paths['mask']}\nCaption: {paths['caption']}")
#             except Exception as e:
#                 st.exception(e)
#
#     if clear_btn:
#         st.session_state.last_pred = None
#         st.rerun()
# else:
#     st.info("Upload both Image A and Image B to proceed.")