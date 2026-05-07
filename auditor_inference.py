#!/usr/bin/env python3
"""
Adversarial Image Auditor (ResNet101 Backbone)
----------------------------------------------
A multi-task evaluation suite for detecting safety violations (Nudity/Violence)
and adversarial artifacts in Text-to-Image generations.

This script includes the full model architecture and a high-level API 
for evaluating local image files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os
import cv2
import json
import urllib.request
import tempfile
import matplotlib.pyplot as plt
from typing import Union, Dict, Any

# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

CLASS_NAMES = ['Safe', 'Nudity', 'Violence']
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_URL = "https://huggingface.co/kricko/Adversarial-Image-Auditor-v2/resolve/main/complete_auditor_best.pth"
DEFAULT_VOCAB_URL = "https://huggingface.co/kricko/Adversarial-Image-Auditor-v2/resolve/main/vocab.json"


def _to_hf_resolve_url(url: str) -> str:
    """Convert Hugging Face blob links to direct resolve links."""
    if not url:
        return url
    return url.replace("/blob/", "/resolve/")


def ensure_artifact(local_path: str, download_url: str, artifact_name: str) -> str:
    """
    Ensure artifact exists locally. Download only once if missing.
    """
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    resolved_url = _to_hf_resolve_url(download_url)
    print(f"[INFO] {artifact_name} not found at '{local_path}'. Downloading from: {resolved_url}")

    fd, tmp_path = tempfile.mkstemp(prefix="download_", suffix=".tmp")
    os.close(fd)
    try:
        urllib.request.urlretrieve(resolved_url, tmp_path)
        if os.path.getsize(tmp_path) == 0:
            raise RuntimeError(f"Downloaded empty file for {artifact_name}.")
        os.replace(tmp_path, local_path)
        print(f"[INFO] Saved {artifact_name} to '{local_path}'.")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return local_path

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class SimpleTokenizer:
    """Word-level tokenizer with padding and SOS/EOS tokens."""
    def __init__(self, vocab_path: str = None, max_length: int = 77):
        self.max_length = max_length
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                self.word_to_idx = json.load(f)
        
    def encode(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(self.max_length, dtype=torch.long)
        words = str(text).lower().split()
        indices = [2] # <SOS>
        for word in words[:self.max_length-2]:
            indices.append(self.word_to_idx.get(word, 1))
        indices.append(3) # <EOS>
        while len(indices) < self.max_length:
            indices.append(0)
        return torch.tensor(indices[:self.max_length], dtype=torch.long)

class SimpleTextEncoder(nn.Module):
    """LSTM-based text encoder for prompt embedding."""
    def __init__(self, vocab_size: int, embed_dim=512, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 512)
        self.norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text_tokens):
        embedded = self.dropout(self.embedding(text_tokens))
        out, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)
        text_features = self.fc(hidden)
        seq_features = self.norm(self.fc(out))
        return text_features, seq_features

class CompleteMultiTaskAuditor(nn.Module):
    """
    Multi-task model with ResNet101 backbone.
    Detects: Binary Adversarial artifacts, Safety categories, Seam Quality, 
    and Prompt Faithfulness.
    """
    def __init__(self, num_classes=3, vocab_size=50000):
        super().__init__()
        resnet = models.resnet101(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.text_encoder = SimpleTextEncoder(vocab_size=vocab_size)
        
        # Detection Heads
        self.adv_head    = nn.Conv2d(2048, 1, kernel_size=1)
        self.class_head  = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.image_proj  = nn.Conv2d(2048, 512, kernel_size=1)
        
        # Cross-Attention
        self.cross_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        self.query_norm = nn.LayerNorm(512); self.key_norm = nn.LayerNorm(512)
        
        # Projection for CLIP-style faithfulness
        self.img_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        self.txt_proj_head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256))
        
        # Stability and Artifact heads
        self.timestep_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 512))
        self.film_adv  = nn.Linear(512, 2048 * 2)
        self.relative_adv_head = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 1))
        self.seam_feat = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(512))
        self.seam_cls = nn.Sequential(nn.Conv2d(512, 1, kernel_size=1))

    def forward(self, x, text_tokens=None, timestep=None):
        B = x.size(0)
        feats = self.features(x)
        global_feats = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        
        # 1. Direct Visual Classifiers
        adv_map = self.adv_head(feats)
        class_map = self.class_head(feats)
        
        # 2. Text-Conditioned Path
        if text_tokens is not None:
            text_feat, seq_feat = self.text_encoder(text_tokens)
            img_seq = self.image_proj(feats).view(B, 512, -1).permute(0, 2, 1)
            att_seq, _ = self.cross_attention(self.query_norm(img_seq), self.key_norm(seq_feat), self.key_norm(seq_feat))
            img_embed = F.normalize(self.img_proj_head(att_seq.mean(dim=1)), dim=-1)
            txt_embed = F.normalize(self.txt_proj_head(text_feat), dim=-1)
        else:
            img_embed = txt_embed = None

        # 3. Artifact Scoring
        if timestep is not None:
            ts_feat = self.timestep_embed(timestep)
            gamma, beta = self.film_adv(ts_feat).chunk(2, dim=-1)
            global_mod = (1.0 + gamma) * global_feats + beta
            rel_score = torch.sigmoid(self.relative_adv_head(global_mod))
        else:
            rel_score = torch.sigmoid(self.relative_adv_head(global_feats))
            
        seam_map = torch.sigmoid(self.seam_cls(self.seam_feat(feats)))
        seam_score = F.adaptive_avg_pool2d(seam_map, (1, 1)).flatten(1)

        return {
            'binary_logits': F.adaptive_avg_pool2d(adv_map, (1, 1)).flatten(1),
            'class_logits':  F.adaptive_avg_pool2d(class_map, (1, 1)).flatten(1),
            'adversarial_map': torch.sigmoid(adv_map),
            'img_embed': img_embed,
            'txt_embed': txt_embed,
            'seam_quality_score': seam_score,
            'relative_adv_score': rel_score
        }

# =============================================================================
# INFERENCE API
# =============================================================================

class AdversarialAuditor:
    """Simplified class for high-level image auditing."""
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        model_url: str = DEFAULT_MODEL_URL,
        vocab_url: str = DEFAULT_VOCAB_URL,
    ):
        model_path = ensure_artifact(model_path, model_url, "model weights")
        vocab_path = ensure_artifact(vocab_path, vocab_url, "vocab")

        self.tokenizer = SimpleTokenizer(vocab_path)
        vocab_size = len(self.tokenizer.word_to_idx)
        
        self.model = CompleteMultiTaskAuditor(num_classes=NUM_CLASSES, vocab_size=vocab_size)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE).eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def audit(self, image_input: Union[str, Image.Image], prompt: str = "") -> Dict[str, Any]:
        """Runs full audit on an image and return structured results."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input.convert("RGB")
            
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        tokens = self.tokenizer.encode(prompt).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(img_tensor, text_tokens=tokens, timestep=torch.zeros(1, 1).to(DEVICE))
            
        prob = torch.sigmoid(outputs['binary_logits']).item()
        class_probs = F.softmax(outputs['class_logits'], dim=1)[0]
        pred_idx = torch.argmax(class_probs).item()
        
        # Calculate faithfulness (cosine similarity)
        cos_sim = F.cosine_similarity(outputs['img_embed'], outputs['txt_embed']).item()
        
        return {
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(class_probs[pred_idx]),
            "is_adversarial": prob > 0.5,
            "adversarial_probability": prob,
            "seam_quality": outputs['seam_quality_score'].item(),
            "faithfulness_score": (cos_sim + 1.0) / 2.0,
            "adversarial_heatmap": outputs['adversarial_map'][0, 0].cpu().numpy()
        }

def save_audit_plot(image_path: str, results: Dict[str, Any], output_path: str):
    image = Image.open(image_path).convert("RGB")
    heatmap = cv2.resize(results['adversarial_heatmap'], (image.size[0], image.size[1]))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original: {results['prediction']} ({results['confidence']:.1%})")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(heatmap, alpha=0.5, cmap='jet')
    plt.title(f"Adversary Map (Prob: {results['adversarial_probability']:.1%})")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adversarial Image Auditor CLI")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", default="", help="Prompt used to generate the image")
    parser.add_argument("--model", default="checkpoints/complete_auditor_best.pth", help="Path to .pth weights")
    parser.add_argument("--vocab", default="checkpoints/vocab.json", help="Path to vocab.json")
    parser.add_argument("--model-url", default=DEFAULT_MODEL_URL, help="HF URL to download weights if missing")
    parser.add_argument("--vocab-url", default=DEFAULT_VOCAB_URL, help="HF URL to download vocab if missing")
    parser.add_argument("--output", default="audit_report.png", help="Path to save visual report")
    args = parser.parse_args()

    auditor = AdversarialAuditor(
        model_path=args.model,
        vocab_path=args.vocab,
        model_url=args.model_url,
        vocab_url=args.vocab_url,
    )
    res = auditor.audit(args.image, args.prompt)
    
    print("\n" + "="*40)
    print("AUDIT RESULTS")
    print("="*40)
    print(f"Safety Class:   {res['prediction']} ({res['confidence']:.1%})")
    print(f"Adversarial:    {res['is_adversarial']} ({res['adversarial_probability']:.1%})")
    print(f"Seam Quality:   {res['seam_quality']:.3f}")
    print(f"Faithfulness:   {res['faithfulness_score']:.3f}")
    print("="*40)
    
    save_audit_plot(args.image, res, args.output)
    print(f"\n[DONE] Visual report saved to: {args.output}")
