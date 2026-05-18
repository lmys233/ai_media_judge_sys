from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 尝试加载环境变量文件
env_path = ROOT / ".env.dev"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=False)
else:
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

from src.decision.gateway import ConfidenceGateway
from src.feature.desensitize import DesensitizeProcessor
from src.models.contracts import ManualAuditCaseMessage, ViolationType
from src.prelabel.text_nb import NaiveBayesTextClassifier

LABEL_ZH = {
    "normal": "正常内容",
    "abuse": "辱骂/攻击性语言",
    "violence": "暴力威胁",
    "porn": "色情不当内容",
    "politics": "政治敏感/误导",
    "other": "其他风险",
}


def _build_components():
    classifier = NaiveBayesTextClassifier()
    desensitizer = DesensitizeProcessor()
    gateway = ConfidenceGateway()
    return classifier, desensitizer, gateway


def run_once(text: str, classifier=None, desensitizer=None, gateway=None) -> dict:
    if classifier is None:
        classifier, desensitizer, gateway = _build_components()
    pred = classifier.predict(text)
    sanitized = desensitizer.sanitize(text)
    route = gateway.route(pred.risk_score)
    return {
        "input_text": text,
        "sanitized_text": sanitized,
        "prelabel": pred.label.value,
        "prelabel_zh": LABEL_ZH.get(pred.label.value, pred.label.value),
        "decision_source": pred.decision_source,
        "risk_score": round(pred.risk_score, 4),
        "evidence": pred.evidence,
        "route": route,
    }


def train_and_predict(train_cases: list[dict], test_texts: list[str]) -> None:
    """Simulate: ingest human-reviewed cases -> retrain NB -> predict new texts."""
    classifier, desensitizer, gateway = _build_components()

    print("=" * 60)
    print("Phase 1: Training NB with human-reviewed cases")
    print("=" * 60)
    for i, case_data in enumerate(train_cases, 1):
        text = case_data["text"]
        label = case_data["label"]
        classifier.append_training_sample(
            text=text, label=label, source="manual_review", verified=True,
        )
        print(f"  [{i}] appended: label={label}  text={text[:40]}...")

    retrained = classifier.retrain_from_corpus()
    print(f"\n  NB retrained: {retrained}")

    print("\n" + "=" * 60)
    print("Phase 2: Desensitize training texts (rule-based + LLM)")
    print("=" * 60)
    for i, case_data in enumerate(train_cases, 1):
        text = case_data["text"]
        labels = [case_data["label"]]
        rule_result = desensitizer.sanitize(text)
        llm_result = desensitizer.sanitize_for_rag(text, labels)
        print(f"  [{i}] original:     {text}")
        print(f"       rule-based:   {rule_result}")
        print(f"       llm-rewrite:  {llm_result}")
        print()

    print("=" * 60)
    print("Phase 3: Predict new texts with retrained NB")
    print("=" * 60)
    for i, text in enumerate(test_texts, 1):
        result = run_once(text, classifier, desensitizer, gateway)
        print(f"  [{i}] {json.dumps(result, ensure_ascii=False)}")

    print("\nDone.")


def interactive_mode() -> None:
    classifier, desensitizer, gateway = _build_components()
    print("Text moderation demo (interactive). Commands:")
    print("  train <label> <text>  - add a training sample and retrain")
    print("  <text>                - predict")
    print("  exit                  - quit")
    while True:
        line = input("\n> ").strip()
        if not line:
            continue
        if line.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if line.lower().startswith("train "):
            parts = line.split(None, 2)
            if len(parts) < 3:
                print("Usage: train <label> <text>")
                continue
            label, text = parts[1], parts[2]
            classifier.append_training_sample(text=text, label=label, source="interactive", verified=True)
            retrained = classifier.retrain_from_corpus()
            print(f"  Appended & retrained={retrained}, label={label}")
            continue
        result = run_once(line, classifier, desensitizer, gateway)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Text moderation demo: predict, train, or run full cycle.")
    parser.add_argument("--text", type=str, default="", help="Single text for one-shot prediction")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--demo-cycle", action="store_true", help="Run a full train-then-predict demo cycle")
    args = parser.parse_args()

    if args.demo_cycle:
        train_cases = [
            {"text": "你这个废物 赶紧去死吧 垃圾", "label": "abuse"},
            {"text": "我要举报你散布政治谣言", "label": "politics"},
            {"text": "免费成人视频 点击观看", "label": "porn"},
            {"text": "再骚扰我就砍死你全家", "label": "violence"},
            {"text": "今天的课程非常有趣，学到了很多", "label": "normal"},
            {"text": "滚你妈的 老子弄死你", "label": "abuse"},
        ]
        test_texts = [
            "你真是个废物 滚远点",
            "今天阳光明媚心情好",
            "色情网站 免费注册",
            "再惹我就弄死你",
            "选举舞弊 政治阴谋",
        ]
        train_and_predict(train_cases, test_texts)
        return

    if args.interactive:
        interactive_mode()
        return

    if not args.text:
        parser.error("Use --text, --interactive, or --demo-cycle.")

    result = run_once(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
