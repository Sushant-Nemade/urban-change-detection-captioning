"""Adapted from Change-Agent; see THIRD_PARTY_NOTICES.md. State-dict keys are retained."""

import torch
from einops import rearrange
from torch import nn


class Encoder(nn.Module):
    """Shared SegFormer B1 encoder; parameter names match published checkpoints."""

    def __init__(self, network="segformer-mit_b1", backbone_weights=None):
        super().__init__()
        if network != "segformer-mit_b1":
            raise ValueError("The released dual-head model supports segformer-mit_b1")
        self.network = network
        from .segformer import Segformer_baseline

        self.cnn = Segformer_baseline(backbone="mit_b1", pretrained=backbone_weights)

    def forward(self, imageA, imageB):
        feat1_list, feat1 = self.cnn.segformer.stage_123(imageA)
        feat2_list, feat2 = self.cnn.segformer.stage_123(imageB)
        feat1_CD = self.cnn.segformer.stage_4(feat1)
        feat2_CD = self.cnn.segformer.stage_4(feat2)
        feat1_list.extend([feat1_CD, feat1_CD])
        feat2_list.extend([feat2_CD, feat2_CD])
        return (feat1_list, feat2_list)

    def fine_tune(self, fine_tune=True):
        for parameter in self.parameters():
            parameter.requires_grad = fine_tune


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Dynamic_conv(nn.Module):
    def __init__(self, dim):
        super(Dynamic_conv, self).__init__()
        self.d_conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.d_conv_1x5 = nn.Conv2d(dim, dim, kernel_size=(1, 5), padding=(0, 2), groups=dim)
        self.d_conv_5x1 = nn.Conv2d(dim, dim, kernel_size=(5, 1), padding=(2, 0), groups=dim)
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(3 * dim)
        self.conv_1 = nn.Conv2d(3 * dim, dim, 1)

    def forward(self, x):
        x1 = self.d_conv_3x3(x)
        x2 = self.d_conv_1x5(x)
        x3 = self.d_conv_5x1(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.BN(x)
        x = self.activation(x)
        x = self.conv_1(x)
        return x


class MultiHeadAtt(nn.Module):
    def __init__(self, dim_q, dim_kv, attention_dim, heads=8, dropout=0.0):
        super(MultiHeadAtt, self).__init__()
        project_out = not (heads == 1 and attention_dim == dim_kv)
        self.heads = heads
        self.scale = (attention_dim // self.heads) ** (-0.5)
        self.to_q = nn.Linear(dim_q, attention_dim, bias=True)
        self.to_k = nn.Linear(dim_kv, attention_dim, bias=True)
        self.to_v = nn.Linear(dim_kv, attention_dim, bias=True)
        self.Q_LN = nn.LayerNorm(dim_q)
        self.K_LN = nn.LayerNorm(dim_kv)
        self.V_LN = nn.LayerNorm(dim_kv)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = (
            nn.Sequential(nn.Linear(attention_dim, dim_q), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(1 * dim_kv, dim_kv, 1), nn.BatchNorm2d(dim_kv), nn.ReLU()
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(2 * dim_kv, dim_kv, 1), nn.BatchNorm2d(dim_kv), nn.ReLU()
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x1, x2, x3):
        batch, L, c = x1.shape
        cross = not x1.equal(x2)
        h = torch.sqrt(torch.tensor(L).float()).int()
        w = torch.sqrt(torch.tensor(L).float()).int()
        x1_feat = x1.transpose(-1, 1).view(batch, c, h, w)
        x2_feat = x2.transpose(-1, 1).view(batch, c, h, w)
        x3_feat = x3.transpose(-1, 1).view(batch, c, h, w)
        x1_feat_buff = x1_feat
        if cross:
            dif = x2_feat - x1_feat
            x2_feat = self.fuse_conv(x1_feat * dif)
            x3_feat = x2_feat
        x1 = x1_feat.view(batch, c, -1).transpose(-1, 1)
        x2 = x2_feat.view(batch, c, -1).transpose(-1, 1)
        x3 = x3_feat.view(batch, c, -1).transpose(-1, 1)
        x1_feat_buff = x1_feat_buff.view(batch, c, -1).transpose(-1, 1)
        q = self.to_q(x1)
        k = self.to_k(x2)
        v = self.to_v(x3)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.dropout(self.attend(dots))
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout=0.0, norm_first=False
    ):
        super(Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)
        self.Q_d_conv = Dynamic_conv(dim_q)
        self.K_d_conv = Dynamic_conv(dim_kv)
        group = dim_q
        self.PCM = nn.Sequential(
            nn.Conv2d(dim_q, dim_q, kernel_size=(3, 3), stride=1, padding=(1, 1), groups=group),
            nn.BatchNorm2d(dim_q),
            nn.GELU(),
            nn.Conv2d(dim_q, dim_q, kernel_size=(1, 1), stride=1),
        )

    def forward(self, x1, x2, x3):
        batch, L, c = x1.shape
        h = torch.sqrt(torch.tensor(L).float()).int()
        w = torch.sqrt(torch.tensor(L).float()).int()
        x1_feat = x1.transpose(-1, 1).view(batch, c, h, w)
        x2_feat = x2.transpose(-1, 1).view(batch, c, h, w)
        x3_feat = x3.transpose(-1, 1).view(batch, c, h, w)
        x1_feat = x1_feat + self.Q_d_conv(x1_feat)
        x2_feat = x2_feat + self.K_d_conv(x2_feat)
        x3_feat = x3_feat + self.K_d_conv(x3_feat)
        x1 = x1_feat.view(batch, c, -1).transpose(-1, 1)
        x2 = x2_feat.view(batch, c, -1).transpose(-1, 1)
        x3 = x3_feat.view(batch, c, -1).transpose(-1, 1)
        res = x1_feat.view(batch, c, -1).transpose(-1, 1)
        if self.norm_first:
            x = self.att(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + res
            x = self.feedforward(self.norm2(x)) + x
        else:
            x = self.norm1(self.att(x1, x2, x3) + res)
            x = self.norm2(self.feedforward(x) + x)
        return x


class Q_Transformer(nn.Module):
    def __init__(
        self, dim_q, dim_kv, heads, attention_dim, hidden_dim, dropout=0.0, norm_first=False
    ):
        super(Q_Transformer, self).__init__()
        self.norm_first = norm_first
        self.att = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.att2 = MultiHeadAtt(dim_q, dim_kv, attention_dim, heads=heads, dropout=dropout)
        self.feedforward = FeedForward(dim_q, hidden_dim=4 * dim_q, dropout=dropout)
        self.norm0 = nn.LayerNorm(dim_q)
        self.norm1 = nn.LayerNorm(dim_q)
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, x1, x2, x3):
        if self.norm_first:
            x1 = self.att(self.norm0(x1), self.norm0(x1), self.norm0(x1)) + x1
            x = self.att2(self.norm1(x1), self.norm1(x2), self.norm1(x3)) + x1
            x = self.feedforward(self.norm2(x)) + x
        else:
            x1 = self.norm0(self.att(x1, x1, x1) + x1)
            x = self.norm1(self.att(x1, x2, x3) + x1)
            x = self.norm2(self.feedforward(x) + x)
        return x


class AttentiveEncoder(nn.Module):
    """
    Visual transformer neck with two branches: captioning and change detection.
    """

    def __init__(self, train_stage, n_layers, feature_size, heads, dropout=0.0):
        super(AttentiveEncoder, self).__init__()
        h_feat, w_feat, channels = feature_size
        self.train_stage = train_stage
        self.h_embedding = nn.Embedding(h_feat, int(channels / 2))
        self.w_embedding = nn.Embedding(w_feat, int(channels / 2))
        self.Dynamic_DIF_aware_TR = nn.ModuleList([])
        for i in range(n_layers):
            self.Dynamic_DIF_aware_TR.append(
                nn.ModuleList(
                    [
                        Transformer(
                            dim_q=channels,
                            dim_kv=channels,
                            heads=heads,
                            attention_dim=channels,
                            hidden_dim=4 * channels,
                            dropout=dropout,
                            norm_first=False,
                        ),
                        Transformer(
                            dim_q=channels,
                            dim_kv=channels,
                            heads=heads,
                            attention_dim=channels,
                            hidden_dim=4 * channels,
                            dropout=dropout,
                            norm_first=False,
                        ),
                        nn.Linear(channels * 2, channels),
                    ]
                )
            )
        self.cap_modules_list = [self.h_embedding, self.w_embedding, self.Dynamic_DIF_aware_TR]
        self.h_embedding_CD = nn.Embedding(h_feat, int(channels / 2))
        self.w_embedding_CD = nn.Embedding(w_feat, int(channels / 2))
        dims = [64, 128, 320, 512]
        self.Transformer_aug_CD = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Transformer(
                            dim_q=dims[-1],
                            dim_kv=dims[-1],
                            heads=heads,
                            attention_dim=dims[-1],
                            hidden_dim=4 * dims[-1],
                            dropout=dropout,
                            norm_first=False,
                        ),
                        Transformer(
                            dim_q=dims[-1],
                            dim_kv=dims[-1],
                            heads=heads,
                            attention_dim=dims[-1],
                            hidden_dim=4 * dims[-1],
                            dropout=dropout,
                            norm_first=False,
                        ),
                    ]
                )
                for _ in range(3)
            ]
        )
        self.conv_dif = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim)) for dim in dims]
        )
        self.conv_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(3 * dim, 2 * dim, 3, stride=1, padding=1),
                    nn.BatchNorm2d(2 * dim),
                    nn.ReLU(),
                    nn.Conv2d(2 * dim, 2 * dim, 1),
                )
                for dim in dims
            ]
        )
        self.cos = torch.nn.CosineSimilarity(dim=1)
        self.to_fused = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2 * dim, 2 * dim, 1),
                    nn.BatchNorm2d(2 * dim),
                    nn.ReLU(),
                    nn.ConvTranspose2d(dim * 2, 2 * dims[max(i - 1, 0)], 4, stride=2, padding=1),
                )
                for i, dim in enumerate(dims)
            ]
        )
        num_classes = 3
        self.to_seg = nn.Sequential(
            nn.ConvTranspose2d(dims[0] * 2, dims[0], 4, stride=2, padding=1),
            nn.Conv2d(int(dims[0]), num_classes, 1),
        )
        self.CD_modules_list = [
            self.Transformer_aug_CD,
            self.conv_dif,
            self.conv_fuse,
            self.cos,
            self.to_fused,
            self.to_seg,
            self.h_embedding_CD,
            self.w_embedding_CD,
        ]
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def add_pos_embedding(self, x):
        batch, c, h, w = x.shape
        device = x.device
        pos_h = torch.arange(h, device=device)
        pos_w = torch.arange(w, device=device)
        embed_h = self.w_embedding(pos_h)
        embed_w = self.h_embedding(pos_w)
        pos_embedding = torch.cat(
            [embed_w.unsqueeze(0).repeat(h, 1, 1), embed_h.unsqueeze(1).repeat(1, w, 1)], dim=-1
        )
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        return x + pos_embedding

    def add_pos_embedding_CD(self, x):
        batch, c, h, w = x.shape
        device = x.device
        pos_h = torch.arange(h, device=device)
        pos_w = torch.arange(w, device=device)
        embed_h = self.w_embedding_CD(pos_h)
        embed_w = self.h_embedding_CD(pos_w)
        pos_embedding = torch.cat(
            [embed_w.unsqueeze(0).repeat(h, 1, 1), embed_h.unsqueeze(1).repeat(1, w, 1)], dim=-1
        )
        pos_embedding = pos_embedding.permute(2, 0, 1).unsqueeze(0).repeat(batch, 1, 1, 1)
        return x + pos_embedding

    def prepare_caption(self, img1, img2, CD_feat_list=None):
        batch, c, h, w = img1.shape
        img1 = img1.view(batch, c, -1).transpose(-1, 1)
        img2 = img2.view(batch, c, -1).transpose(-1, 1)
        img_sa1, img_sa2 = (img1, img2)
        for layer_a, layer_b, linear in self.Dynamic_DIF_aware_TR:
            img_sa1_tr1 = layer_a(img_sa1, img_sa2, img_sa2)
            img_sa2_tr1 = layer_b(img_sa2, img_sa1, img_sa1)
            img_sa1 = img_sa1_tr1
            img_sa2 = img_sa2_tr1
        img1 = img_sa1_tr1.transpose(-1, 1).view(batch, c, h, w)
        img2 = img_sa2_tr1.transpose(-1, 1).view(batch, c, h, w)
        feat_list = [img1, img2]
        return (img1, img2, feat_list)

    def change_detection(self, img1_list, img2_list, CC_feat_list=None):
        feat_num = len(img1_list)
        img_fus_list = []
        feat_1_last = img1_list[-1]
        feat_2_last = img2_list[-1]
        b, n, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, n, -1).transpose(-1, 1)
        feat_2_last = feat_2_last.view(b, n, -1).transpose(-1, 1)
        feat_1_last = feat_1_last.transpose(-1, 1).view(b, n, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, n, h, w)
        img1_list[-1] = feat_1_last
        img2_list[-1] = feat_2_last
        for k in range(feat_num):
            dif = self.conv_dif[k](img2_list[k] - img1_list[k]) + self.cos(
                img1_list[k], img2_list[k]
            ).unsqueeze(1)
            fus = torch.cat([img1_list[k], dif, img2_list[k]], dim=1)
            fus = self.conv_fuse[k](fus)
            img_fus_list.append(fus)
        up = self.to_fused[-1](img_fus_list[-1])
        for i in range(feat_num - 1, 0, -1):
            i = i - 1
            img_fus = img_fus_list[i] + up
            img_fus = self.to_fused[i](img_fus)
            up = img_fus
        seg = self.to_seg(img_fus)
        return seg

    def CC_neck_s0(self, feat_1_last, feat_2_last):
        feat_1_last = self.add_pos_embedding(feat_1_last)
        feat_2_last = self.add_pos_embedding(feat_2_last)
        b, c, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, c, -1).transpose(-1, 1)
        feat_2_last = feat_2_last.view(b, c, -1).transpose(-1, 1)
        feat_1_last = feat_1_last.transpose(-1, 1).view(b, c, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, c, h, w)
        feat = [feat_1_last, feat_2_last]
        return (feat_1_last, feat_2_last, feat)

    def CD_neck_s0(self, img1_list, img2_list):
        feat_1_last = self.add_pos_embedding_CD(img1_list[-1])
        feat_2_last = self.add_pos_embedding_CD(img2_list[-1])
        b, c, h, w = feat_1_last.size()
        feat_1_last = feat_1_last.view(b, c, -1).transpose(-1, 1)
        feat_2_last = feat_2_last.view(b, c, -1).transpose(-1, 1)
        for layerA, layerB in self.Transformer_aug_CD:
            feat_1_last = layerA(feat_1_last, feat_2_last, feat_2_last) + feat_1_last
            feat_2_last = layerB(feat_2_last, feat_1_last, feat_1_last) + feat_2_last
        feat_1_last = feat_1_last.transpose(-1, 1).view(b, c, h, w)
        feat_2_last = feat_2_last.transpose(-1, 1).view(b, c, h, w)
        img1_list[-1] = feat_1_last
        img2_list[-1] = feat_2_last
        feat = [img1_list[-1], img2_list[-1]]
        return (img1_list, img2_list, feat)

    def forward(self, img1_list, img2_list):
        CD_img1_list, CD_img2_list = (img1_list[:4], img2_list[:4])
        CC_img1_feat, CC_img2_feat = (img1_list[-1], img2_list[-1])
        CD_img1_list, CD_img2_list, _ = self.CD_neck_s0(CD_img1_list, CD_img2_list)
        CC_img1_feat, CC_img2_feat, _ = self.CC_neck_s0(CC_img1_feat, CC_img2_feat)
        img1_cap, img2_cap, _ = self.prepare_caption(CC_img1_feat, CC_img2_feat, None)
        seg = self.change_detection(CD_img1_list, CD_img2_list, None)
        return (img1_cap, img2_cap, seg)

    def fine_tune(self, goal, fine_tune=True):
        """
        Control which submodules are trainable:
          goal=0 -> detection-only   (CD modules train; caption modules frozen/eval)
          goal=1 -> caption-only     (caption modules train; CD modules frozen/eval)
          goal=2 -> joint            (both branches train)
        """
        assert goal in [0, 1, 2], f"Unsupported goal={goal}"
        for p in self.parameters():
            p.requires_grad = False

        def _train_modules(mods, flag=True):
            for m in mods:
                m.train(flag)
                for p in m.parameters():
                    p.requires_grad = flag

        if goal == 0:
            _train_modules(self.CD_modules_list, True)
            _train_modules(self.cap_modules_list, False)
        elif goal == 1:
            _train_modules(self.cap_modules_list, True)
            _train_modules(self.CD_modules_list, False)
        else:
            _train_modules(self.cap_modules_list, True)
            _train_modules(self.CD_modules_list, True)
