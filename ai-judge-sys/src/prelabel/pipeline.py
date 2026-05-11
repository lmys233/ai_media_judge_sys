from __future__ import annotations

import logging

from src.models.contracts import MediaType, PreLabelResult, ViolationType
from src.prelabel.text_nb import NaiveBayesTextClassifier

logger = logging.getLogger(__name__)


class PreLabelPipeline:
    def __init__(self) -> None:
        self.text_classifier = NaiveBayesTextClassifier()
        self._image_classifier = None
        self._video_summarizer = None

    @property
    def image_classifier(self):
        """Lazy load image classifier to avoid torch import failure on Windows."""
        if self._image_classifier is None:
            from src.prelabel.image_mobilenet import MobileNetImageClassifier
            self._image_classifier = MobileNetImageClassifier()
        return self._image_classifier

    @property
    def video_summarizer(self):
        """Lazy load video summarizer to avoid torch import failure on Windows."""
        if self._video_summarizer is None:
            from src.prelabel.video_summary import VideoFrameSummarizer
            self._video_summarizer = VideoFrameSummarizer()
        return self._video_summarizer

    def run(self, parsed: dict) -> tuple[PreLabelResult, str]:
        media_type = parsed["media_type"]
        if media_type == MediaType.TEXT.value:
            pred = self.text_classifier.predict(parsed.get("content", ""))
            return (
                PreLabelResult(
                    violation_candidates=[pred.label],
                    risk_score_pre=pred.risk_score,
                    evidence=pred.evidence,
                    model_version="text-nb-v1",
                ),
                f"文本初筛结果: {pred.label.value}, 风险分={pred.risk_score:.2f}",
            )

        if media_type == MediaType.IMAGE.value:
            pred = self.image_classifier.predict(parsed["media_path"])
            return (
                PreLabelResult(
                    violation_candidates=[pred.label],
                    risk_score_pre=pred.risk_score,
                    evidence=pred.evidence,
                    model_version="image-mobilenet-v1",
                ),
                f"图片初筛结果: {pred.label.value}, 风险分={pred.risk_score:.2f}",
            )

        frame_paths = parsed.get("frame_paths", [])
        labels: list[ViolationType] = []
        scores: list[float] = []
        frame_evidence: list[str] = []
        for frame_path in frame_paths:
            pred = self.image_classifier.predict(frame_path)
            labels.append(pred.label)
            scores.append(pred.risk_score)
            frame_evidence.extend(pred.evidence)

        final_label, final_score = self.video_summarizer.infer_label(labels, scores)
        summary = self.video_summarizer.summarize(frame_evidence)
        result = PreLabelResult(
            violation_candidates=[final_label],
            risk_score_pre=final_score,
            evidence=frame_evidence[:20],
            model_version="video-frame-mobilenet-v1",
        )
        return result, summary
