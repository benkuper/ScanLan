from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


class ModelPack(str, Enum):
    COMMERCIAL = "commercial"
    RESEARCH = "research"


@dataclass(frozen=True)
class ModelCandidate:
    identifier: str
    roles: tuple[str, ...]
    source_url: str
    source_revision: str
    model_url: str | None
    model_revision: str | None
    code_license: str
    model_license: str
    commercial_use: bool
    output_restricted: bool
    status: str
    notes: str


# P13 freezes candidates, revisions and legal policy, but does not bundle these
# material weights yet. P14 may integrate only a candidate that passes the
# representative ScanLan bake-off and whose complete dependency tree matches
# the requested pack.
MODEL_CANDIDATES = (
    ModelCandidate(
        "material-anything-estimator",
        ("pbr-estimation", "albedo", "roughness", "metallic", "normal"),
        "https://github.com/3DTopia/MaterialAnything",
        "be3d6b32a195f968540abc2ee106dc02d4b07479",
        "https://huggingface.co/xanderhuang/material_estimator",
        "dcd4e4c213363f351f642c0e3551771bc03d1c4c",
        "MIT",
        "Apache-2.0",
        True,
        False,
        "bakeoff",
        "Strong released mesh-conditioned PBR candidate; full transitive asset audit remains mandatory.",
    ),
    ModelCandidate(
        "rgb-to-x",
        ("intrinsic-decomposition", "albedo", "roughness", "metallic", "lighting"),
        "https://github.com/zheng95z/rgbx",
        "977e0df27d369d3e68900399f59a42b0156d4440",
        "https://huggingface.co/zheng95z/rgb-to-x",
        "b38b3fd73a14ea62f3953fc54bc4ac67b067bae0",
        "Adobe-Research-License",
        "UNVERIFIED",
        False,
        True,
        "bakeoff",
        "Interior-scene intrinsic challenger; research-only source and unverified checkpoint terms.",
    ),
    ModelCandidate(
        "diffusion-renderer-inverse",
        ("intrinsic-decomposition", "video", "lighting", "material"),
        "https://github.com/nv-tlabs/diffusion-renderer",
        "8fcf0057ad3422139cd53281037025ff725d34e9",
        None,
        None,
        "NVIDIA-NonCommercial",
        "NVIDIA-NonCommercial",
        False,
        True,
        "bakeoff",
        "High-quality video inverse-rendering challenger; license permits research/evaluation only.",
    ),
)


def resolve_model_pack(
    pack: ModelPack | str,
    candidates: Iterable[ModelCandidate] = MODEL_CANDIDATES,
) -> tuple[ModelCandidate, ...]:
    requested = ModelPack(pack)
    values = tuple(candidates)
    identifiers = [candidate.identifier for candidate in values]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model candidate identifiers must be unique")
    for candidate in values:
        if not candidate.source_revision or len(candidate.source_revision) != 40:
            raise ValueError(f"{candidate.identifier} does not pin a source revision")
        if candidate.model_url and (
            not candidate.model_revision or len(candidate.model_revision) != 40
        ):
            raise ValueError(f"{candidate.identifier} does not pin its model revision")
        if candidate.model_license == "UNVERIFIED" and candidate.commercial_use:
            raise ValueError(f"{candidate.identifier} cannot be commercial with unverified terms")
    if requested is ModelPack.COMMERCIAL:
        return tuple(
            candidate
            for candidate in values
            if candidate.commercial_use and not candidate.output_restricted
        )
    return values


def write_pack_manifest(
    path: Path,
    pack: ModelPack | str,
    candidates: Iterable[ModelCandidate] = MODEL_CANDIDATES,
) -> dict[str, Any]:
    requested = ModelPack(pack)
    included = resolve_model_pack(requested, candidates)
    manifest = {
        "schemaVersion": 1,
        "pack": requested.value,
        "commercialUse": requested is ModelPack.COMMERCIAL,
        "outputRestrictions": sorted(
            {candidate.model_license for candidate in included if candidate.output_restricted}
        ),
        "models": [asdict(candidate) for candidate in included],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest
