# Training and fine-tuning

Prepare and validate the dataset before training. Input images and indexed/grayscale labels must be 256 × 256. Configurations are plain JSON; command-line arguments override configuration fields.

## Stage one: joint initialization

```bash
urban-change download backbone --output artifacts/backbones/mit_b1.pth
urban-change train --config configs/train_joint.json --device cuda
```

The recipe uses batch size 8 and 8 accumulation steps. Adjust these to measured GPU memory. `--random-init` explicitly permits training without pretrained backbone weights, primarily for controlled scratch experiments. Inference from a full MCI checkpoint does not require this backbone file.

## Stage two: specialize and optionally reunite

```bash
urban-change train --config configs/finetune_caption.json --device cuda

urban-change train --config configs/finetune_caption.json \
  --goal detection --checkpoint runs/s2-caption/best.pth --output runs/s2-detection

urban-change train --config configs/finetune_caption.json \
  --goal joint --loss-balance --checkpoint runs/s2-caption/best.pth --output runs/s2-joint
```

Joint training updates both heads and the encoder. Detection-only training freezes caption-specific modules and the decoder. Caption fine-tuning freezes the encoder; `--unfreeze-caption-neck` selects whether the caption neck can adapt. Validation always switches the entire model to evaluation mode, and training reapplies the selected freeze states afterward.

When moving between vocabularies, new checkpoints carry the source mapping. For a custom legacy snapshot, pass `--source-vocab`. Shared token rows are transplanted by token identity; unknown new tokens keep their initialization. Equal vocabulary sizes alone are insufficient to establish compatible mappings.

## Resume a run

```bash
urban-change train --config configs/train_joint.json \
  --resume runs/s1-joint/last.pth --epochs 50
```

`--resume` ignores an initialization checkpoint inherited from the JSON recipe; explicitly passing both `--checkpoint` and `--resume` is rejected. Resume restores model, optimizer, scheduler, AMP scaler, epoch, best metric, and Python/NumPy/PyTorch RNG states. Changes to optimization, vocabulary, or model settings are rejected. Epoch count, worker count, device, and output location may change, but bitwise reproducibility across devices or worker settings is not promised.

Each run writes `config.json`, `history.jsonl`, `last.pth`, and the best `best.pth`. A nonempty directory requires resume rather than silently combining experiments. Best-model selection uses validation BLEU-4 for captions, mean IoU for detection, and their sum for joint training. Test results never select the best checkpoint.

Accumulation flushes the final partial window and normalizes by its actual microbatch count. Small datasets therefore still execute an optimizer step. Mixed precision is enabled only on CUDA; CPU training is supported for validation and small experiments. Full GPU training has not been benchmarked in this CPU environment.
