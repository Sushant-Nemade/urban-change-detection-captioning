"""Build the original model graph without coupling it to filesystem paths."""

import re
from dataclasses import asdict, dataclass

from torch import nn

from urban_change.models.decoder import DecoderTransformer
from urban_change.models.encoder import AttentiveEncoder, Encoder


@dataclass(frozen=True)
class ModelConfig:
    encoder_dim: int = 512
    feature_dim: int = 512
    feature_size: int = 16
    heads: int = 8
    neck_layers: int = 3
    decoder_layers: int = 1
    dropout: float = 0.1
    max_length: int = 46


def config_from_checkpoint(checkpoint: dict) -> ModelConfig:
    decoder = checkpoint["decoder_dict"]
    try:
        max_length = int(decoder["position_encoding.pe"].shape[0])
        decoder_layers = (
            max(int(k.split(".")[2]) for k in decoder if re.match(r"transformer\.layers\.\d+\.", k))
            + 1
        )
        neck_layers = (
            max(
                int(k.split(".")[1])
                for k in checkpoint["encoder_trans_dict"]
                if re.match(r"Dynamic_DIF_aware_TR\.\d+\.", k)
            )
            + 1
        )
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError("Checkpoint does not match the supported MCI architecture") from exc
    metadata = dict(checkpoint.get("model_config", {}))
    metadata.update(max_length=max_length, decoder_layers=decoder_layers, neck_layers=neck_layers)
    return ModelConfig(**metadata)


class ChangeModel(nn.Module):
    def __init__(
        self, vocab: dict[str, int], config: ModelConfig | None = None, backbone_weights=None
    ):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.encoder = Encoder("segformer-mit_b1", backbone_weights=backbone_weights)
        self.encoder_trans = AttentiveEncoder(
            None, c.neck_layers, [c.feature_size, c.feature_size, c.encoder_dim], c.heads, c.dropout
        )
        self.decoder = DecoderTransformer(
            c.encoder_dim,
            c.feature_dim,
            len(vocab),
            c.max_length,
            vocab,
            c.heads,
            c.decoder_layers,
            c.dropout,
        )

    def forward(self, image_a, image_b):
        features_a, features_b = self.encoder(image_a, image_b)
        return self.encoder_trans(features_a, features_b)

    def load_legacy_state(self, checkpoint: dict) -> None:
        # Missing neck weights used to be silently accepted. A public inference
        # endpoint must fail rather than return results from random parameters.
        self.encoder.load_state_dict(checkpoint["encoder_dict"], strict=True)
        self.encoder_trans.load_state_dict(checkpoint["encoder_trans_dict"], strict=True)
        self.decoder.load_state_dict(checkpoint["decoder_dict"], strict=True)

    def checkpoint_state(self) -> dict:
        return {
            "encoder_dict": self.encoder.state_dict(),
            "encoder_trans_dict": self.encoder_trans.state_dict(),
            "decoder_dict": self.decoder.state_dict(),
            "model_config": asdict(self.config),
        }
