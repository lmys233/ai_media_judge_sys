from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    import joblib
except Exception:  # noqa: BLE001
    CountVectorizer = None
    MultinomialNB = None
    joblib = None
import yaml

from src.models.contracts import ViolationType

logger = logging.getLogger(__name__)


@dataclass
class TextPrediction:
    label: ViolationType
    risk_score: float
    evidence: list[str]
    decision_source: str = "unknown"


@dataclass
class MultiLabelPrediction:
    """Multi-label prediction result containing predictions for each detected violation type."""
    predictions: list[TextPrediction]
    all_violation_types: list[ViolationType]  # All types detected in text
    top_label: ViolationType  # Highest risk violation type
    overall_risk_score: float  # Overall risk score
    decision_source: str = "multi_label"


class NaiveBayesTextClassifier:
    """PoC text classifier with seed data, replace with real dataset training."""

    # 每类训练样本容量上限（超出后淘汰最旧的非人工样本）
    MAX_PER_CLASS = 2000

    # 按类型分区的文件名映射
    _CLASS_FILE_MAP = {
        "normal": "train_normal.jsonl",
        "abuse": "train_abuse.jsonl",
        "violence": "train_violence.jsonl",
        "porn": "train_porn.jsonl",
        "politics": "train_politics.jsonl",
    }

    def __init__(self) -> None:
        self.use_sklearn = CountVectorizer is not None and MultinomialNB is not None
        self.vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=1) if self.use_sklearn else None
        self.model = MultinomialNB() if self.use_sklearn else None
        self.label_to_violation = {
            0: ViolationType.NORMAL,
            1: ViolationType.ABUSE,
            2: ViolationType.VIOLENCE,
            3: ViolationType.PORN,
            4: ViolationType.POLITICS,
        }
        self.policy_path = Path(__file__).resolve().parent.parent / "policy" / "text_keyword_overrides.yaml"
        self.policy_mtime = 0.0
        self.hard_keywords, self.soft_keywords = self._load_keyword_rules()
        self.data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "text_nb"
        self.seed_path = self.data_dir / "seed_samples.jsonl"
        self.train_path = self.data_dir / "train_samples.jsonl"
        self.model_path = self.data_dir / "nb_model.joblib"
        self.vec_path = self.data_dir / "nb_vectorizer.joblib"
        self.hash_path = self.data_dir / "corpus_hash.txt"
        # 按类型分区的训练文件路径
        self.class_file_map: dict[str, Path] = {
            label: self.data_dir / fname
            for label, fname in self._CLASS_FILE_MAP.items()
        }
        self._trained = False
        self._train_bootstrap()

    def _corpus_hash(self) -> str:
        """Hash of all training data files to detect changes."""
        h = hashlib.sha256()
        # 按类型分区的训练文件
        for class_file in self.class_file_map.values():
            if class_file.exists():
                h.update(class_file.read_bytes())
        return h.hexdigest()

    def _train_bootstrap(self) -> None:
        if not self.use_sklearn:
            self._trained = True
            return

        current_hash = self._corpus_hash()
        if self._try_load_model(current_hash):
            logger.info("NB模型已从磁盘加载（语料未变化）")
            self._trained = True
            return

        texts, labels = self._load_training_corpus()
        x = self.vectorizer.fit_transform(texts)
        self.model.fit(x, labels)
        self._trained = True
        self._save_model(current_hash)
        logger.info("NB模型已从语料训练完成（%d 条样本）并保存到磁盘", len(texts))

    def _try_load_model(self, current_hash: str) -> bool:
        if joblib is None:
            return False
        if not (self.model_path.exists() and self.vec_path.exists() and self.hash_path.exists()):
            return False
        saved_hash = self.hash_path.read_text(encoding="utf-8").strip()
        if saved_hash != current_hash:
            return False
        try:
            self.vectorizer = joblib.load(self.model_path)
            self.model = joblib.load(self.vec_path)
            return True
        except Exception:  # noqa: BLE001
            return False

    def _save_model(self, corpus_hash: str) -> None:
        if joblib is None:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(self.vectorizer, self.model_path)
            joblib.dump(self.model, self.vec_path)
            self.hash_path.write_text(corpus_hash, encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.warning("NB模型保存失败，下次启动将重新训练")

    def predict(self, text: str) -> TextPrediction:
        self._ensure_latest_rules()
        if not text.strip():
            return TextPrediction(label=ViolationType.NORMAL, risk_score=0.0, evidence=[], decision_source="empty")

        keyword_match = self._predict_by_hard_keyword(text)
        if keyword_match is not None:
            soft_hits = self._collect_hits(text, self.soft_keywords)
            if soft_hits:
                extra_terms: list[str] = []
                for values in soft_hits.values():
                    extra_terms.extend(values)
                keyword_match.evidence = list(dict.fromkeys(keyword_match.evidence + extra_terms))[:8]
                keyword_match.risk_score = min(0.99, keyword_match.risk_score + 0.02 * len(set(extra_terms)))
                keyword_match.decision_source = f"{keyword_match.decision_source}_hybrid"
            return keyword_match

        if not self.use_sklearn:
            return self._predict_fallback(text)

        x = self.vectorizer.transform([text])
        probs = self.model.predict_proba(x)[0]
        pred_idx = int(probs.argmax())
        confidence = float(probs[pred_idx])
        label = self.label_to_violation[pred_idx]

        risk_score = confidence if label != ViolationType.NORMAL else 1 - confidence
        evidence = [token for token in text.split()[:8]]
        base = TextPrediction(
            label=label,
            risk_score=max(0.0, min(1.0, risk_score)),
            evidence=evidence,
            decision_source="nb",
        )
        return self._apply_soft_keyword_bonus(text, base)

    def predict_multi(self, text: str) -> MultiLabelPrediction:
        """Predict multiple violation types for a given text.

        Returns a MultiLabelPrediction containing predictions for each detected
        violation type, along with evidence for each type.
        """
        self._ensure_latest_rules()
        if not text.strip():
            return MultiLabelPrediction(
                predictions=[],
                all_violation_types=[],
                top_label=ViolationType.NORMAL,
                overall_risk_score=0.0,
                decision_source="empty",
            )

        # Get all detected violation types from hard keywords
        hard_keyword_types = self._detect_all_hard_keyword_types(text)
        soft_keyword_types = self._detect_all_soft_keyword_types(text)

        # Merge all detected types
        all_detected_types = list(set(hard_keyword_types + soft_keyword_types))

        if not all_detected_types:
            # No keywords detected, use NB model for single prediction
            single_pred = self.predict(text)
            return MultiLabelPrediction(
                predictions=[single_pred],
                all_violation_types=[single_pred.label] if single_pred.label != ViolationType.NORMAL else [],
                top_label=single_pred.label,
                overall_risk_score=single_pred.risk_score,
                decision_source=single_pred.decision_source,
            )

        # Build predictions for each detected type
        predictions: list[TextPrediction] = []
        for vtype in all_detected_types:
            pred = self._predict_single_type(text, vtype)
            predictions.append(pred)

        # Sort by risk score descending
        predictions.sort(key=lambda p: p.risk_score, reverse=True)

        top_pred = predictions[0]
        overall_risk = max(p.risk_score for p in predictions)

        # Determine decision source
        source = "multi_hard" if hard_keyword_types else "multi_soft"
        if len(all_detected_types) > 1:
            source += "_multi_type"

        return MultiLabelPrediction(
            predictions=predictions,
            all_violation_types=all_detected_types,
            top_label=top_pred.label,
            overall_risk_score=overall_risk,
            decision_source=source,
        )

    def _detect_all_hard_keyword_types(self, text: str) -> list[ViolationType]:
        """Detect all violation types via hard keyword matching."""
        lowered = text.lower()
        detected: list[ViolationType] = []
        for key, label in [
            ("violence", ViolationType.VIOLENCE),
            ("porn", ViolationType.PORN),
            ("politics", ViolationType.POLITICS),
            ("abuse", ViolationType.ABUSE),
        ]:
            keywords = self.hard_keywords.get(key, [])
            if any(kw in lowered or kw in text for kw in keywords):
                detected.append(label)
        return detected

    def _detect_all_soft_keyword_types(self, text: str) -> list[ViolationType]:
        """Detect all violation types via soft keyword matching."""
        lowered = text.lower()
        detected: list[ViolationType] = []
        category_map = {
            "abuse": ViolationType.ABUSE,
            "violence": ViolationType.VIOLENCE,
            "porn": ViolationType.PORN,
            "politics": ViolationType.POLITICS,
        }
        for category, label in category_map.items():
            keywords = self.soft_keywords.get(category, [])
            if any(kw in lowered or kw in text for kw in keywords):
                detected.append(label)
        return detected

    def _predict_single_type(self, text: str, target_type: ViolationType) -> TextPrediction:
        """Predict risk score and evidence for a specific violation type."""
        # Collect evidence for this type
        evidence: list[str] = []

        # From hard keywords
        hard_key_map = {
            ViolationType.ABUSE: "abuse",
            ViolationType.VIOLENCE: "violence",
            ViolationType.PORN: "porn",
            ViolationType.POLITICS: "politics",
        }
        hard_key = hard_key_map.get(target_type)
        if hard_key:
            hard_kws = self.hard_keywords.get(hard_key, [])
            hits = [kw for kw in hard_kws if kw in text.lower() or kw in text]
            evidence.extend(hits)

        # From soft keywords
        soft_key = hard_key_map.get(target_type)
        if soft_key:
            soft_kws = self.soft_keywords.get(soft_key, [])
            hits = [kw for kw in soft_kws if kw in text.lower() or kw in text]
            evidence.extend(hits)

        # Calculate risk score based on hits
        hit_count = len(evidence)
        if hit_count == 0:
            # Fallback to NB probability for this type
            if self.use_sklearn:
                x = self.vectorizer.transform([text])
                probs = self.model.predict_proba(x)[0]
                idx_map = {v: i for i, v in self.label_to_violation.items()}
                if target_type in idx_map:
                    prob = float(probs[idx_map[target_type]])
                    risk = prob if target_type != ViolationType.NORMAL else 1 - prob
                    return TextPrediction(
                        label=target_type,
                        risk_score=max(0.0, min(1.0, risk)),
                        evidence=[],
                        decision_source="nb_single",
                    )
            return TextPrediction(
                label=target_type,
                risk_score=0.3,
                evidence=[],
                decision_source="low_confidence",
            )

        # Calculate risk based on keyword hit count
        base_score = 0.6
        score = min(0.99, base_score + 0.08 * hit_count)

        # Boost if hard keyword hit
        if target_type in self._detect_all_hard_keyword_types(text):
            score = min(0.99, score + 0.15)

        return TextPrediction(
            label=target_type,
            risk_score=score,
            evidence=list(dict.fromkeys(evidence))[:8],
            decision_source="keyword_analysis",
        )

    def _predict_fallback(self, text: str) -> TextPrediction:
        rule_map = [
            (ViolationType.VIOLENCE, self.soft_keywords.get("violence", [])),
            (ViolationType.PORN, self.soft_keywords.get("porn", [])),
            (ViolationType.POLITICS, self.soft_keywords.get("politics", [])),
            (ViolationType.ABUSE, self.soft_keywords.get("abuse", [])),
        ]
        lowered = text.lower()
        category_hits: dict[ViolationType, list[str]] = {}
        for label, keywords in rule_map:
            hits = [kw for kw in keywords if kw in lowered or kw in text]
            if hits:
                category_hits[label] = hits
        if category_hits:
            top_label = max(category_hits, key=lambda k: len(category_hits[k]))
            total_hits = sum(len(v) for v in category_hits.values())
            category_count = len(category_hits)
            score = min(0.95, 0.42 + 0.1 * total_hits + 0.06 * (category_count - 1))
            merged_hits: list[str] = []
            for values in category_hits.values():
                merged_hits.extend(values)
            evidence = list(dict.fromkeys(merged_hits))[:8]
            source = "soft_rule_multi" if category_count > 1 else "soft_rule"
            return TextPrediction(label=top_label, risk_score=score, evidence=evidence, decision_source=source)
        return TextPrediction(
            label=ViolationType.NORMAL,
            risk_score=0.1,
            evidence=text.split()[:8],
            decision_source="fallback_normal",
        )

    def _predict_by_hard_keyword(self, text: str) -> TextPrediction | None:
        lowered = text.lower()
        check_order = [
            ("violence", ViolationType.VIOLENCE),
            ("porn", ViolationType.PORN),
            ("politics", ViolationType.POLITICS),
            ("abuse", ViolationType.ABUSE),
        ]
        category_hits: dict[ViolationType, list[str]] = {}
        for key, label in check_order:
            keywords = self.hard_keywords.get(key, [])
            hits = [kw for kw in keywords if kw in lowered or kw in text]
            if hits:
                category_hits[label] = hits
        if category_hits:
            # Priority: violence/porn first when hit counts tie.
            priority = {
                ViolationType.VIOLENCE: 4,
                ViolationType.PORN: 3,
                ViolationType.POLITICS: 2,
                ViolationType.ABUSE: 1,
            }
            top_label = max(category_hits, key=lambda k: (len(category_hits[k]), priority.get(k, 0)))
            total_hits = sum(len(v) for v in category_hits.values())
            category_count = len(category_hits)
            score = min(0.99, 0.7 + 0.07 * total_hits + 0.05 * (category_count - 1))
            merged_hits: list[str] = []
            for values in category_hits.values():
                merged_hits.extend(values)
            evidence = list(dict.fromkeys(merged_hits))[:8]
            source = "hard_rule_multi" if category_count > 1 else "hard_rule"
            return TextPrediction(label=top_label, risk_score=score, evidence=evidence, decision_source=source)
        return None

    def _collect_hits(self, text: str, keywords_by_category: dict[str, list[str]]) -> dict[ViolationType, list[str]]:
        lowered = text.lower()
        category_to_violation = {
            "abuse": ViolationType.ABUSE,
            "violence": ViolationType.VIOLENCE,
            "porn": ViolationType.PORN,
            "politics": ViolationType.POLITICS,
        }
        results: dict[ViolationType, list[str]] = {}
        for category, label in category_to_violation.items():
            words = keywords_by_category.get(category, [])
            hits = [kw for kw in words if kw in lowered or kw in text]
            if hits:
                results[label] = hits
        return results

    def _apply_soft_keyword_bonus(self, text: str, base: TextPrediction) -> TextPrediction:
        match_terms = self._collect_hits(text, self.soft_keywords)
        match_counts: dict[ViolationType, int] = {k: len(v) for k, v in match_terms.items()}

        if not match_counts:
            return base

        top_label = max(match_counts, key=lambda k: match_counts[k])
        hit_count = match_counts[top_label]
        if top_label == base.label:
            boosted = min(0.97, base.risk_score + 0.08 * hit_count)
            evidence = list(dict.fromkeys(base.evidence + match_terms[top_label]))[:8]
            return TextPrediction(label=base.label, risk_score=boosted, evidence=evidence, decision_source="hybrid")

        # Soft signals can correct uncertain NB outputs, but should not override strong NB confidence.
        if base.risk_score < 0.7 and hit_count >= 2:
            override_score = min(0.9, 0.55 + 0.1 * hit_count)
            return TextPrediction(
                label=top_label,
                risk_score=override_score,
                evidence=match_terms[top_label][:8],
                decision_source="soft_override",
            )
        all_soft_terms: list[str] = []
        for values in match_terms.values():
            all_soft_terms.extend(values)
        merged_evidence = list(dict.fromkeys(base.evidence + all_soft_terms))[:8]
        return TextPrediction(
            label=base.label,
            risk_score=base.risk_score,
            evidence=merged_evidence,
            decision_source="nb_soft_signal",
        )

    def _ensure_latest_rules(self) -> None:
        try:
            mtime = self.policy_path.stat().st_mtime if self.policy_path.exists() else 0.0
        except Exception:  # noqa: BLE001
            return
        if mtime > self.policy_mtime:
            self.hard_keywords, self.soft_keywords = self._load_keyword_rules()
            self.policy_mtime = mtime

    def _load_keyword_rules(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        hard_defaults = {
            "abuse": ["操你妈", "你妈死了"],
            "violence": ["把你杀了", "爆你头", "弄死你", "砍死"],
            "porn": ["和你妈做爱", "强奸"],
            "politics": [],
        }
        soft_defaults = {
            "abuse": ["垃圾", "傻逼", "滚开", "辱骂", "狗东西"],
            "violence": ["打死", "砍人", "爆头", "报复", "血腥", "威胁"],
            "porn": ["裸露", "情色", "约炮", "成人视频", "色情", "做爱", "性行为", "裸聊"],
            "politics": ["选举", "党派", "政治", "抗议", "阴谋", "政治谣言"],
        }
        if not self.policy_path.exists():
            return hard_defaults, soft_defaults

        def merge(base: dict[str, list[str]], extra: dict[str, Any]) -> dict[str, list[str]]:
            merged = {k: list(v) for k, v in base.items()}
            for key in merged:
                values = extra.get(key, [])
                if isinstance(values, list):
                    merged[key] = list(dict.fromkeys(merged[key] + [str(i) for i in values]))
            return merged

        try:
            payload = yaml.safe_load(self.policy_path.read_text(encoding="utf-8")) or {}
            # Backward compatibility: old format uses `keyword_overrides`.
            legacy = payload.get("keyword_overrides", {})
            hard_section = payload.get("hard_keywords", {})
            soft_section = payload.get("soft_keywords", {})
            if isinstance(legacy, dict):
                soft_section = {**legacy, **soft_section}

            hard = merge(hard_defaults, hard_section if isinstance(hard_section, dict) else {})
            soft = merge(soft_defaults, soft_section if isinstance(soft_section, dict) else {})
        except Exception:  # noqa: BLE001
            return hard_defaults, soft_defaults
        return hard, soft

    def append_training_sample(
        self,
        text: str,
        label: str,
        source: str = "manual_review",
        reviewer_id: str = "",
        verified: bool = True,
    ) -> None:
        if label not in {"normal", "abuse", "violence", "porn", "politics"}:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "text": text,
            "label": label,
            "source": source,
            "reviewer_id": reviewer_id,
            "verified": verified,
            "updated_at": datetime.utcnow().isoformat(),
        }
        # 写入该类型的分区文件
        class_file = self.class_file_map[label]
        with class_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 追加后检查该类型是否超限，超限则裁剪
        self._trim_class_file(label)

    def _trim_class_file(self, label: str) -> None:
        """裁剪单个类型的训练文件至 MAX_PER_CLASS，优先淘汰最旧的非人工样本。"""
        class_file = self.class_file_map.get(label)
        if not class_file or not class_file.exists():
            return

        lines = class_file.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.MAX_PER_CLASS:
            return

        records: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if len(records) <= self.MAX_PER_CLASS:
            return

        # 分离人工验证样本和自动标注样本
        verified = [r for r in records if r.get("verified")]
        auto = [r for r in records if not r.get("verified")]

        # 按时间排序（旧的在前）
        auto.sort(key=lambda r: r.get("updated_at", ""))

        # 保留全部人工样本 + 最新的自动样本
        keep_verified = len(verified)
        keep_auto = max(0, self.MAX_PER_CLASS - keep_verified)

        if keep_auto >= len(auto):
            return  # 无需裁剪

        kept = verified + (auto[-keep_auto:] if keep_auto > 0 else [])
        kept.sort(key=lambda r: r.get("updated_at", ""))

        new_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n"
        class_file.write_text(new_text, encoding="utf-8")

        logger.info(
            "NB训练样本已裁剪: label=%s, before=%d, after=%d (verified=%d, auto=%d)",
            label, len(records), len(kept), keep_verified, keep_auto,
        )

    def _trim_all_classes(self) -> None:
        """重训练前统一裁剪所有类型，确保容量合规。"""
        for label in self.class_file_map:
            self._trim_class_file(label)

    def retrain_from_corpus(self) -> bool:
        if not self.use_sklearn:
            return False
        # 重训练前先裁剪各类型，防止无限膨胀
        self._trim_all_classes()
        texts, labels = self._load_training_corpus()
        if not texts:
            return False
        x = self.vectorizer.fit_transform(texts)
        self.model.fit(x, labels)
        self._trained = True
        self._save_model(self._corpus_hash())
        logger.info("NB模型已重新训练（%d 条样本）并保存到磁盘", len(texts))
        return True

    def _load_training_corpus(self) -> tuple[list[str], list[int]]:
        label_to_idx = {
            "normal": 0,
            "abuse": 1,
            "violence": 2,
            "porn": 3,
            "politics": 4,
        }
        built_in = [
            ("今天天气很好 我们一起学习", "normal"),
            ("你好 世界 这是一个正常内容", "normal"),
            ("你这个人真垃圾 滚开", "abuse"),
            ("操你妈 脏话 攻击他人", "abuse"),
            ("我要打死你 报复 砍人", "violence"),
            ("血腥暴力 打架视频", "violence"),
            ("裸露 色情 约炮", "porn"),
            ("成人视频 情色 内容", "porn"),
            ("政治敏感 选举 党派 争议", "politics"),
            ("抗议集会 政治冲突", "politics"),
        ]
        records: list[tuple[str, str]] = list(built_in)

        def _read_jsonl(path: Path) -> list[tuple[str, str]]:
            items: list[tuple[str, str]] = []
            if not path.exists():
                return items
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(row.get("text", "")).strip()
                label = str(row.get("label", "")).strip()
                if text and label in label_to_idx:
                    items.append((text, label))
            return items

        # 从按类型分区的文件中读取
        for class_label, class_file_path in self.class_file_map.items():
            if class_file_path.exists():
                for line in class_file_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(row.get("text", "")).strip()
                    row_label = str(row.get("label", "")).strip()
                    if text and row_label in label_to_idx:
                        records.append((text, row_label))

        # 旧版单文件(seed_samples.jsonl / train_samples.jsonl)已迁移完毕不再读取

        texts: list[str] = []
        labels: list[int] = []
        for text, label in records:
            texts.append(text)
            labels.append(label_to_idx[label])
        return texts, labels
