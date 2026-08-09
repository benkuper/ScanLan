from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


COLMAP_ALIKED_MODEL = "aliked-n16rot.onnx"
COLMAP_LIGHTGLUE_MODEL = "aliked-lightglue.onnx"


def _runtime_model(filename: str, environment_name: str) -> Path:
    configured = os.environ.get(environment_name)
    executable_root = Path(sys.executable).resolve().parent
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend(
        (
            executable_root / "models" / filename,
            executable_root.parent / "models" / filename,
            Path(__file__).resolve().parent / "models" / filename,
            Path(__file__).resolve().parent.parent / "models" / filename,
        )
    )
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root) / "models" / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"{filename} is not installed; run npm run prepare:splat or set {environment_name}"
    )


@lru_cache(maxsize=1)
def pycolmap_feature_runtime() -> dict[str, Any]:
    """Validate that CUDA SIFT and learned extraction/matching actually execute.

    A CUDA-compiled module is not sufficient evidence that its native DLLs,
    driver, and selected compute architecture work on the current machine.
    Keep backend selection explicit and cache the smoke test for this process.
    """
    # Importing Torch on Windows registers its bundled CUDA/cuDNN DLLs. The
    # independently built ONNX Runtime used by COLMAP's learned features needs
    # those DLLs to be loaded before its CUDA execution provider is created.
    import torch

    import pycolmap

    result: dict[str, Any] = {
        "version": str(pycolmap.__version__),
        "cudaBuilt": bool(pycolmap.has_cuda),
        "cudaValidated": False,
        "backend": "COLMAP SIFT CPU",
        "featureCount": 0,
        "matchCount": 0,
        "learnedValidated": False,
        "featureType": "SIFT",
        "matcherType": "SIFT_BRUTEFORCE",
        "alikedModel": None,
        "lightglueModel": None,
        "learnedError": None,
        "error": None,
    }
    if not pycolmap.has_cuda:
        result["error"] = "PyCOLMAP was compiled without CUDA"
        return result
    if not torch.cuda.is_available():
        result["error"] = "PyTorch cannot initialize CUDA for PyCOLMAP"
        return result
    try:
        # Deterministic noise supplies stable gradients without depending on a
        # test asset or filesystem access in the packaged worker.
        image = (
            np.random.default_rng(0x5CA11A).random((384, 384)) * 255.0
        ).astype(np.uint8)
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.max_image_size = 384
        extraction.sift.max_num_features = 512
        extractor = pycolmap.FeatureExtractor.create(
            extraction,
            pycolmap.Device.cuda,
        )
        keypoints, descriptors = extractor.extract_from_uint8_array(image)
        matcher = pycolmap.FeatureMatcher.create(
            pycolmap.FeatureMatchingOptions(),
            pycolmap.Device.cuda,
        )
        matches = matcher.match(
            keypoints,
            descriptors,
            keypoints,
            descriptors,
        )
        feature_count = len(keypoints)
        match_count = int(matches.shape[0])
        if feature_count < 16 or match_count < 8:
            raise RuntimeError(
                f"CUDA SIFT smoke test returned only {feature_count} features and {match_count} matches"
            )
        result.update(
            cudaValidated=True,
            backend="COLMAP CUDA SIFT",
            featureCount=feature_count,
            matchCount=match_count,
            error=None,
        )
    except Exception as error:
        result["error"] = str(error)
        return result

    # COLMAP documents ALIKED + LightGlue as its higher-quality learned path
    # for challenging viewpoint and illumination changes. Validate the native
    # ONNX/CUDA execution and exact packaged models before selecting it; SIFT
    # remains a fully working fallback if the learned runtime cannot execute.
    try:
        aliked_model = _runtime_model(
            COLMAP_ALIKED_MODEL,
            "SCANLAN_COLMAP_ALIKED_MODEL",
        )
        lightglue_model = _runtime_model(
            COLMAP_LIGHTGLUE_MODEL,
            "SCANLAN_COLMAP_LIGHTGLUE_MODEL",
        )
        import cv2

        rgb_image = np.repeat(image[..., None], 3, axis=2)
        for y in range(32, 352, 64):
            for x in range(32, 352, 64):
                cv2.circle(rgb_image, (x, y), 11, (255, 255, 255), 2)
        translated_image = cv2.warpAffine(
            rgb_image,
            np.float32(((1.0, 0.0, 4.0), (0.0, 1.0, 3.0))),
            (384, 384),
            borderMode=cv2.BORDER_REFLECT,
        )
        learned_extraction = pycolmap.FeatureExtractionOptions()
        learned_extraction.type = pycolmap.FeatureExtractorType.ALIKED_N16ROT
        learned_extraction.max_image_size = 384
        learned_extraction.aliked.max_num_features = 512
        learned_extraction.aliked.n16rot_model_path = str(aliked_model)
        learned_matching = pycolmap.FeatureMatchingOptions()
        learned_matching.type = pycolmap.FeatureMatcherType.ALIKED_LIGHTGLUE
        learned_matching.max_num_matches = 4_096
        learned_matching.aliked.lightglue.model_path = str(lightglue_model)
        # LightGlue normalizes keypoints with each image's calibrated camera.
        # The low-level PyCOLMAP match() binding omits that camera metadata, so
        # validate the same database-backed path used by ScanLan's real SfM.
        with tempfile.TemporaryDirectory(
            prefix="scanlan-lightglue-",
            ignore_cleanup_errors=True,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            image_root = temporary_root / "images"
            image_root.mkdir()
            if not cv2.imwrite(str(image_root / "000.png"), rgb_image):
                raise RuntimeError("could not write the first learned-feature smoke image")
            if not cv2.imwrite(str(image_root / "001.png"), translated_image):
                raise RuntimeError("could not write the second learned-feature smoke image")
            database_path = temporary_root / "database.db"
            pycolmap.extract_features(
                database_path,
                image_root,
                camera_mode=pycolmap.CameraMode.SINGLE,
                extraction_options=learned_extraction,
                device=pycolmap.Device.cuda,
            )
            pycolmap.match_exhaustive(
                database_path,
                matching_options=learned_matching,
                device=pycolmap.Device.cuda,
            )
            with sqlite3.connect(database_path) as database:
                learned_feature_count = int(
                    database.execute(
                        "SELECT COALESCE(SUM(rows), 0) FROM keypoints"
                    ).fetchone()[0]
                )
                learned_match_count = int(
                    database.execute(
                        "SELECT COALESCE(SUM(rows), 0) FROM matches"
                    ).fetchone()[0]
                )
                learned_inlier_count = int(
                    database.execute(
                        "SELECT COALESCE(SUM(rows), 0) FROM two_view_geometries"
                    ).fetchone()[0]
                )
        if learned_feature_count < 16 or learned_match_count < 8:
            raise RuntimeError(
                "learned feature smoke test returned only "
                f"{learned_feature_count} features and {learned_match_count} matches"
            )
        if learned_inlier_count < 8:
            raise RuntimeError(
                "learned feature smoke test geometrically verified only "
                f"{learned_inlier_count} inliers"
            )
        result.update(
            backend="COLMAP CUDA ALIKED + LightGlue",
            featureCount=learned_feature_count,
            matchCount=learned_match_count,
            inlierCount=learned_inlier_count,
            learnedValidated=True,
            featureType="ALIKED_N16ROT",
            matcherType="ALIKED_LIGHTGLUE",
            alikedModel=str(aliked_model),
            lightglueModel=str(lightglue_model),
            learnedError=None,
        )
    except Exception as error:
        result["learnedError"] = str(error)
    return result


def pycolmap_device(pycolmap: Any) -> Any:
    runtime = pycolmap_feature_runtime()
    return pycolmap.Device.cuda if runtime["cudaValidated"] else pycolmap.Device.cpu
