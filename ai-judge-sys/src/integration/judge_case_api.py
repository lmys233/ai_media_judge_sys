from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class JudgeCaseApiClient:
    """Batch fetch case details from Java backend."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "http://127.0.0.1:8080"
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def batch_get_cases(self, case_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Batch fetch cases by IDs. Returns dict mapping case_id -> case_data."""
        if not case_ids:
            return {}

        try:
            resp = self._session.post(
                f"{self.base_url}/judge/batch",
                json={"caseIds": [int(cid) for cid in case_ids]},
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning("Batch get cases failed: status=%d", resp.status_code)
                return {}

            data = resp.json()
            if data.get("code") != 200 or not data.get("data"):
                return {}

            result: dict[str, dict[str, Any]] = {}
            for case in data["data"]:
                case_id = str(case.get("caseId"))
                # Parse JSON fields
                case_data = dict(case)
                try:
                    case_data["aiResult"] = (
                        case.get("aiResult")
                        if isinstance(case.get("aiResult"), dict)
                        else (case.get("aiResult") and eval(case.get("aiResult")))  # noqa: BLE001
                    )
                except Exception:
                    case_data["aiResult"] = None
                result[case_id] = case_data
            return result

        except Exception:  # noqa: BLE001
            logger.exception("Failed to batch fetch cases: case_ids=%s", case_ids)
            return {}
