from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RELEASE_MATRIX_VERSION = 1
RELEASE_EVIDENCE_VERSION = 1


class ReleaseMatrixError(ValueError):
    """Raised when release evidence is malformed or default promotion is unsafe."""


def packaged_release_requirements_path() -> Path:
    return Path(__file__).with_name("release-matrix-requirements.json")


def _load_json(path: Path) -> Any:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def load_release_requirements(path: Path | None = None) -> dict[str, Any]:
    value = _load_json(path or packaged_release_requirements_path())
    if int(value.get("schemaVersion", 0)) != RELEASE_MATRIX_VERSION:
        raise ReleaseMatrixError(
            f"Release requirements must use schema {RELEASE_MATRIX_VERSION}"
        )
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ReleaseMatrixError("Release requirements need at least one scenario")
    identifiers = [str(item.get("scenarioId", "")) for item in scenarios]
    if any(not identifier for identifier in identifiers):
        raise ReleaseMatrixError("Every release scenario needs a scenarioId")
    if len(identifiers) != len(set(identifiers)):
        raise ReleaseMatrixError("Release scenario IDs must be unique")
    return value


def load_release_evidence(paths: Iterable[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        value = _load_json(path)
        candidates = value.get("records") if isinstance(value, Mapping) else None
        if candidates is None:
            candidates = [value]
        if not isinstance(candidates, list):
            raise ReleaseMatrixError(f"Evidence records in {path} must be a list")
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise ReleaseMatrixError(f"Evidence record in {path} must be an object")
            record = dict(candidate)
            if int(record.get("schemaVersion", 0)) != RELEASE_EVIDENCE_VERSION:
                raise ReleaseMatrixError(
                    f"Evidence in {path} must use schema {RELEASE_EVIDENCE_VERSION}"
                )
            record["_evidencePath"] = str(path.resolve(strict=True))
            record["_displayPath"] = path.as_posix()
            records.append(record)
    return records


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _nested(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return None
        current = current[component]
    return current


def _canonical_digest(value: Mapping[str, Any]) -> str:
    normalized = {key: item for key, item in value.items() if not key.startswith("_")}
    payload = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_reasons(
    evidence: Mapping[str, Any],
    required_labels: Sequence[str],
    *,
    verify_artifacts: bool,
) -> list[str]:
    reasons: list[str] = []
    artifacts = evidence.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return ["artifact evidence must be an object"]
    if required_labels and not verify_artifacts:
        reasons.append("artifact byte verification was disabled for this diagnostic run")
    evidence_path = Path(str(evidence.get("_evidencePath", "."))).resolve()
    for label in required_labels:
        entry = artifacts.get(label)
        if not isinstance(entry, Mapping):
            reasons.append(f"required artifact {label} is missing")
            continue
        expected = str(entry.get("sha256", "")).lower()
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            reasons.append(f"artifact {label} has no valid SHA-256 digest")
            continue
        if not verify_artifacts:
            continue
        artifact_path = Path(str(entry.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = evidence_path.parent / artifact_path
        try:
            artifact_path = artifact_path.resolve(strict=True)
            digest = _file_digest(artifact_path)
        except OSError as error:
            reasons.append(f"artifact {label} cannot be read: {error}")
            continue
        if digest != expected:
            reasons.append(f"artifact {label} digest does not match the inspected result")
    return reasons


def _metric_reasons(
    evidence: Mapping[str, Any], requirements: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    metrics = evidence.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return ["metrics must be an object"]
    for name, bounds in requirements.items():
        if not isinstance(bounds, Mapping):
            raise ReleaseMatrixError(f"Metric requirement {name} must be an object")
        value = _finite(_nested(metrics, str(name)))
        if value is None:
            reasons.append(f"required metric {name} is missing or non-finite")
            continue
        minimum = _finite(bounds.get("minimum"))
        maximum = _finite(bounds.get("maximum"))
        if minimum is not None and value < minimum:
            reasons.append(f"metric {name}={value:g} is below the {minimum:g} minimum")
        if maximum is not None and value > maximum:
            reasons.append(f"metric {name}={value:g} exceeds the {maximum:g} maximum")
    return reasons


def _matching_reasons(
    evidence: Mapping[str, Any], requirements: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    for path, expected in requirements.items():
        actual = _nested(evidence, str(path))
        allowed = list(expected) if isinstance(expected, list) else [expected]
        if actual not in allowed:
            reasons.append(
                f"evidence {path}={actual!r} does not match required value(s) {allowed!r}"
            )
    return reasons


def _visual_reasons(evidence: Mapping[str, Any], required: bool) -> list[str]:
    if not required:
        return []
    visual = evidence.get("visualInspection")
    if not isinstance(visual, Mapping):
        return ["independent final-artifact visual inspection is missing"]
    reasons: list[str] = []
    if visual.get("passed") is not True:
        reasons.append("final-artifact visual inspection did not pass")
    if not str(visual.get("reviewer", "")).strip():
        reasons.append("visual inspection reviewer is missing")
    if not str(visual.get("inspectedAt", "")).strip():
        reasons.append("visual inspection timestamp is missing")
    inspected_digest = str(visual.get("artifactSha256", "")).strip().lower()
    if not inspected_digest:
        reasons.append("visual inspection is not bound to an artifact digest")
    else:
        artifacts = evidence.get("artifacts", {})
        artifact_digests = (
            {
                str(item.get("sha256", "")).strip().lower()
                for item in artifacts.values()
                if isinstance(item, Mapping)
            }
            if isinstance(artifacts, Mapping)
            else set()
        )
        if inspected_digest not in artifact_digests:
            reasons.append("visual inspection digest does not identify a declared final artifact")
    return reasons


def _assess_scenario(
    requirement: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    *,
    verify_artifacts: bool,
) -> dict[str, Any]:
    scenario_id = str(requirement["scenarioId"])
    if evidence is None:
        return {
            "scenarioId": scenario_id,
            "category": str(requirement.get("category", "")),
            "accepted": False,
            "status": "missing",
            "reasons": ["no evidence record was supplied"],
            "evidenceDigest": None,
        }
    reasons: list[str] = []
    status = str(evidence.get("status", "not-run"))
    if status != "passed":
        reasons.append(f"evidence status is {status}, not passed")
    if evidence.get("realInput") is not True:
        reasons.append("scenario was not run on representative real input")
    reasons.extend(_matching_reasons(evidence, requirement.get("matches", {})))

    gates = evidence.get("gates", {})
    if not isinstance(gates, Mapping):
        reasons.append("release gates must be an object")
        gates = {}
    for gate in requirement.get("requiredGates", ()):
        if gates.get(str(gate)) is not True:
            reasons.append(f"required release gate {gate} did not pass")
    reasons.extend(_metric_reasons(evidence, requirement.get("requiredMetrics", {})))
    reasons.extend(_visual_reasons(evidence, bool(requirement.get("visualInspection", False))))
    reasons.extend(
        _artifact_reasons(
            evidence,
            [str(label) for label in requirement.get("requiredArtifacts", ())],
            verify_artifacts=verify_artifacts,
        )
    )
    return {
        "scenarioId": scenario_id,
        "category": str(requirement.get("category", "")),
        "accepted": not reasons,
        "status": status,
        "reasons": reasons,
        "evidenceDigest": _canonical_digest(evidence),
        "evidencePath": evidence.get("_displayPath"),
    }


def evaluate_release_matrix(
    *,
    requirements: Mapping[str, Any],
    evidence_records: Sequence[Mapping[str, Any]],
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Evaluate P19 evidence without allowing one scene to stand in for another.

    Every required scenario is independent and fail-closed. Automated metrics, explicit gates,
    content digests, representative real input, and visual review are all preserved as distinct
    evidence so a successful process or unit test cannot be promoted as release quality.
    """

    if int(requirements.get("schemaVersion", 0)) != RELEASE_MATRIX_VERSION:
        raise ReleaseMatrixError(
            f"Release requirements must use schema {RELEASE_MATRIX_VERSION}"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for record in evidence_records:
        scenario_id = str(record.get("scenarioId", ""))
        if not scenario_id:
            raise ReleaseMatrixError("Every evidence record needs a scenarioId")
        if scenario_id in indexed:
            duplicates.add(scenario_id)
        indexed[scenario_id] = record
    if duplicates:
        raise ReleaseMatrixError(
            "Duplicate evidence records are ambiguous: " + ", ".join(sorted(duplicates))
        )

    assessments = [
        _assess_scenario(
            requirement,
            indexed.get(str(requirement["scenarioId"])),
            verify_artifacts=verify_artifacts,
        )
        for requirement in requirements.get("scenarios", ())
    ]
    unknown = sorted(set(indexed) - {item["scenarioId"] for item in assessments})
    complete = bool(assessments) and all(item["accepted"] for item in assessments)
    accepted = [item["scenarioId"] for item in assessments if item["accepted"]]
    blocked = [item["scenarioId"] for item in assessments if not item["accepted"]]
    return {
        "schemaVersion": RELEASE_MATRIX_VERSION,
        "kind": "scanlan-v2-release-matrix-report",
        "evaluatedAt": datetime.now(timezone.utc).isoformat(),
        "requirementsRevision": str(requirements.get("revision", "unknown")),
        "artifactDigestsVerified": bool(verify_artifacts),
        "complete": complete,
        "defaultPromotion": {
            "eligible": complete,
            "selected": dict(requirements.get("candidateDefaults", {})) if complete else {},
            "reason": (
                "Every required real-input scenario and quality gate passed."
                if complete
                else "Defaults remain unchanged until every required scenario passes."
            ),
        },
        "summary": {
            "requiredScenarioCount": len(assessments),
            "acceptedScenarioCount": len(accepted),
            "acceptedScenarios": accepted,
            "blockedScenarios": blocked,
            "unknownEvidenceScenarios": unknown,
        },
        "scenarios": assessments,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_release_matrix_report(path: Path, report: Mapping[str, Any]) -> None:
    _atomic_json(path.resolve(), report)


def write_default_promotion(path: Path, report: Mapping[str, Any]) -> None:
    promotion = report.get("defaultPromotion", {})
    if not isinstance(promotion, Mapping) or promotion.get("eligible") is not True:
        raise ReleaseMatrixError(
            "Default promotion is forbidden because the complete P19 matrix did not pass"
        )
    _atomic_json(
        path.resolve(),
        {
            "schemaVersion": RELEASE_MATRIX_VERSION,
            "kind": "scanlan-default-backend-promotion",
            "requirementsRevision": report.get("requirementsRevision"),
            "promotedAt": datetime.now(timezone.utc).isoformat(),
            "defaults": dict(promotion.get("selected", {})),
        },
    )
