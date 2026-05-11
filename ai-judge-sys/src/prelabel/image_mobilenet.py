from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.models.contracts import ViolationType


@dataclass
class ImagePrediction:
    label: ViolationType
    risk_score: float
    evidence: list[str]


class MobileNetImageClassifier:
    """Image pre-label classifier using MobileNet + keyword mapping."""

    def __init__(self) -> None:
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            weights = MobileNet_V3_Small_Weights.DEFAULT
            self.model = mobilenet_v3_small(weights=weights).to(self.device).eval()
            self.preprocess = weights.transforms()
            self.labels = weights.meta.get("categories", [])
            self.ready = True
        except Exception:  # noqa: BLE001
            self.model = None
            self.preprocess = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
            self.labels = []
            self.ready = False

    def predict(self, image_path: str) -> ImagePrediction:
        image = Image.open(image_path).convert("RGB")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)

        if not self.ready or self.model is None:
            filename = Path(image_path).name.lower()
            if "violence" in filename or "blood" in filename:
                return ImagePrediction(ViolationType.VIOLENCE, 0.65, ["filename_heuristic"])
            if "nsfw" in filename or "porn" in filename:
                return ImagePrediction(ViolationType.PORN, 0.65, ["filename_heuristic"])
            return ImagePrediction(ViolationType.NORMAL, 0.2, ["fallback_no_model"])

        with torch.no_grad():
            logits = self.model(batch)[0]
            probs = torch.softmax(logits, dim=0)
            score, idx = torch.max(probs, dim=0)
        score_float = float(score.item())
        category = self.labels[int(idx.item())].lower() if self.labels else "unknown"

        label, risk = self._map_to_violation(category, score_float)
        return ImagePrediction(label=label, risk_score=risk, evidence=[category])

    def _map_to_violation(self, category: str, confidence: float) -> tuple[ViolationType, float]:
        porn_words = {"bikini", "swimsuit", "bra", "maillot"}
        violence_words = {"rifle", "revolver", "gun", "knife", "projectile", "missile"}

        if any(word in category for word in porn_words):
            return ViolationType.PORN, max(0.55, confidence)
        if any(word in category for word in violence_words):
            return ViolationType.VIOLENCE, max(0.55, confidence)
        return ViolationType.NORMAL, 1 - confidence
