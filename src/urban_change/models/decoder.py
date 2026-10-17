"""Adapted from Change-Agent; see THIRD_PARTY_NOTICES.md. State-dict keys are retained."""

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class resblock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(resblock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, int(outchannel / 2), kernel_size=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 2), int(outchannel / 2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(outchannel / 2)),
            nn.ReLU(),
            nn.Conv2d(int(outchannel / 2), outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x
        out = out + residual
        return F.relu(out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)
        self.embedding_1D = nn.Embedding(52, int(d_model))

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Mesh_TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        layer_norm_eps=1e-05,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super(Mesh_TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(int(d_model), nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        self_att_tgt = self.norm1(tgt + self._sa_block(tgt, tgt_mask, tgt_key_padding_mask))
        enc_att, att_weight = self._mha_block(
            self_att_tgt, memory, memory_mask, memory_key_padding_mask
        )
        x = self.norm2(self_att_tgt + enc_att)
        x = self.norm3(x + self._ff_block(x))
        return x + tgt

    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        x, att_weight = self.multihead_attn(
            x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=True
        )
        return (self.dropout2(x), att_weight)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class StackTransformer(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(StackTransformer, self).__init__()
        self.layers = torch.nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class DecoderTransformer(nn.Module):
    def __init__(
        self,
        encoder_dim,
        feature_dim,
        vocab_size,
        max_lengths,
        word_vocab,
        n_head,
        n_layers,
        dropout,
    ):
        super(DecoderTransformer, self).__init__()
        self.feature_dim = feature_dim
        self.embed_dim = feature_dim
        self.vocab_size = vocab_size
        self.max_lengths = max_lengths
        self.word_vocab = word_vocab
        self.dropout = dropout
        self.Conv1 = nn.Conv2d(encoder_dim * 2, feature_dim, kernel_size=1)
        self.LN = resblock(feature_dim, feature_dim)
        self.vocab_embedding = nn.Embedding(vocab_size, self.embed_dim)
        decoder_layer = Mesh_TransformerDecoderLayer(
            feature_dim, n_head, dim_feedforward=feature_dim * 4, dropout=self.dropout
        )
        self.transformer = StackTransformer(decoder_layer, n_layers)
        self.position_encoding = PositionalEncoding(feature_dim, max_len=max_lengths)
        self.wdc = nn.Linear(feature_dim, vocab_size)
        self.wdc.weight = self.vocab_embedding.weight
        assert self.wdc.weight.data_ptr() == self.vocab_embedding.weight.data_ptr(), (
            "Weight tying failed"
        )
        self.dropout = nn.Dropout(p=self.dropout)
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.init_weights()

    def init_weights(self):
        self.vocab_embedding.weight.data.uniform_(-0.1, 0.1)
        self.wdc.bias.data.fill_(0)

    def forward(self, x1, x2, encoded_captions, caption_lengths):
        x = torch.cat([x1, x2], dim=1)
        x = self.LN(self.Conv1(x))
        batch, channel = (x.size(0), x.size(1))
        x = x.view(batch, channel, -1).permute(2, 0, 1)
        word_length = encoded_captions.size(1)
        mask = torch.triu(
            torch.ones(word_length, word_length, device=encoded_captions.device) * float("-inf"),
            diagonal=1,
        )
        tgt_pad_mask = (encoded_captions == self.word_vocab["<NULL>"]) | (
            encoded_captions == self.word_vocab["<END>"]
        )
        word_emb = self.vocab_embedding(encoded_captions)
        word_emb = word_emb.transpose(1, 0)
        word_emb = self.position_encoding(word_emb)
        pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
        pred = self.wdc(self.dropout(pred))
        pred = pred.permute(1, 0, 2)
        caption_lengths, sort_ind = caption_lengths.sort(dim=0, descending=True)
        encoded_captions = encoded_captions[sort_ind]
        pred = pred[sort_ind]
        decode_lengths = (caption_lengths - 1).tolist()
        return (pred, encoded_captions, decode_lengths, sort_ind)

    def sample(self, x1, x2, k=1):
        x = torch.cat([x1, x2], dim=1)
        x = self.LN(self.Conv1(x))
        batch, channel = (x.size(0), x.size(1))
        x = x.view(batch, channel, -1).permute(2, 0, 1)
        tgt = torch.zeros(batch, self.max_lengths, dtype=torch.int64, device=x.device)
        mask = torch.triu(
            torch.ones(self.max_lengths, self.max_lengths, device=x.device) * float("-inf"),
            diagonal=1,
        )
        start_id = self.word_vocab["<START>"]
        end_id = self.word_vocab["<END>"]
        tgt[:, 0] = start_id
        seqs = torch.full((batch, 1), start_id, dtype=torch.int64, device=x.device)
        for step in range(self.max_lengths):
            tgt_pad_mask = tgt == self.word_vocab["<NULL>"]
            word_emb = self.vocab_embedding(tgt).transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask, tgt_key_padding_mask=tgt_pad_mask)
            pred = self.wdc(self.dropout(pred))
            scores = pred.permute(1, 0, 2)[:, step, :]
            predicted_id = torch.argmax(scores, dim=-1)
            seqs = torch.cat([seqs, predicted_id.unsqueeze(1)], dim=1)
            if (predicted_id == end_id).all():
                break
            if step < self.max_lengths - 1:
                tgt[:, step + 1] = predicted_id
        seqs = seqs.tolist()
        return seqs[0] if len(seqs) == 1 else seqs

    def sample_beam(self, x1, x2, k=1):
        x = torch.cat([x1, x2], dim=1)
        x = self.LN(self.Conv1(x))
        batch, channel, h, w = x.shape
        assert batch == 1, "batch size must be 1 for beam search"
        x = (
            x.view(batch, channel, -1)
            .unsqueeze(0)
            .expand(k, -1, -1, -1)
            .reshape(batch * k, channel, h * w)
            .permute(2, 0, 1)
        )
        tgt = torch.zeros(k * batch, self.max_lengths, dtype=torch.int64, device=x.device)
        mask = (
            torch.triu(torch.ones(self.max_lengths, self.max_lengths, device=x.device)) == 1
        ).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        start_id = self.word_vocab["<START>"]
        end_id = self.word_vocab["<END>"]
        tgt[:, 0] = start_id
        seqs = torch.full((batch * k, 1), start_id, dtype=torch.int64, device=x.device)
        top_k_scores = torch.zeros(k * batch, 1, device=x.device)
        complete_seqs = []
        complete_seqs_scores = []
        for step in range(self.max_lengths):
            word_emb = self.vocab_embedding(tgt).transpose(1, 0)
            word_emb = self.position_encoding(word_emb)
            pred = self.transformer(word_emb, x, tgt_mask=mask)
            pred = self.wdc(self.dropout(pred))
            scores = pred.permute(1, 0, 2)[:, step, :]
            scores = F.log_softmax(scores, dim=1)
            scores = top_k_scores.expand_as(scores) + scores
            if step == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
            else:
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode="floor")
            next_word_inds = top_k_words % self.vocab_size
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            incomplete_inds = [
                ind for ind, next_word in enumerate(next_word_inds) if next_word != end_id
            ]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            x = x[:, prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            tgt = tgt[incomplete_inds]
            if step < self.max_lengths - 1:
                tgt[:, : step + 2] = seqs
        if complete_seqs == []:
            complete_seqs.extend(seqs.tolist())
            complete_seqs_scores.extend(top_k_scores)
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        return seq

    def fine_tune(self, fine_tune=True):
        for p in self.parameters():
            p.requires_grad = fine_tune
