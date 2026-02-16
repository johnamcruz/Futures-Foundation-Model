"""
Example: Fine-Tuning FFM for Opening Range Breakout (ORB) Strategy

Shows how to load pretrained backbone, attach a classification head,
freeze layers, and run inference.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from futures_foundation import FFMConfig, FFMForClassification


def main():
    print("=" * 60)
    print("  FFM Fine-Tuning Example: ORB Strategy")
    print("=" * 60)

    config = FFMConfig(hidden_size=256, num_hidden_layers=6, num_attention_heads=8)
    model = FFMForClassification(config, num_labels=3)  # BUY=0, SELL=1, HOLD=2

    # Uncomment to load pretrained weights:
    # model.load_backbone("checkpoints/pretrained/best_backbone.pt")
    model.freeze_backbone(freeze_ratio=0.66)

    # --- Inference demo ---
    batch_size, seq_len = 4, 64
    dummy_features = torch.randn(batch_size, seq_len, config.num_features)

    with torch.no_grad():
        outputs = model(features=dummy_features)

    probs = torch.softmax(outputs["logits"], dim=-1)
    predictions = outputs["logits"].argmax(dim=-1)
    label_names = ["BUY", "SELL", "HOLD"]

    for i in range(batch_size):
        pred = label_names[predictions[i]]
        print(f"  Sample {i}: {pred} — BUY={probs[i][0]:.3f} SELL={probs[i][1]:.3f} HOLD={probs[i][2]:.3f}")

    # --- Training demo ---
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    dummy_labels = torch.randint(0, 3, (batch_size,))
    outputs = model(features=dummy_features, labels=dummy_labels)
    outputs["loss"].backward()
    optimizer.step()
    print(f"\n  Training loss: {outputs['loss'].item():.4f}")

    # --- Embedding extraction ---
    with torch.no_grad():
        embeddings = model(features=dummy_features)["embedding"]
    print(f"  Embedding shape: {embeddings.shape}")
    print("\nDone!")

if __name__ == "__main__":
    main()