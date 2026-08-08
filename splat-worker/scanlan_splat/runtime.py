from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np


@lru_cache(maxsize=1)
def pycolmap_feature_runtime() -> dict[str, Any]:
    """Validate that CUDA SIFT extraction and matching actually execute.

    A CUDA-compiled module is not sufficient evidence that its native DLLs,
    driver, and selected compute architecture work on the current machine.
    Keep backend selection explicit and cache the smoke test for this process.
    """
    import pycolmap

    result: dict[str, Any] = {
        "version": str(pycolmap.__version__),
        "cudaBuilt": bool(pycolmap.has_cuda),
        "cudaValidated": False,
        "backend": "COLMAP SIFT CPU",
        "featureCount": 0,
        "matchCount": 0,
        "error": None,
    }
    if not pycolmap.has_cuda:
        result["error"] = "PyCOLMAP was compiled without CUDA"
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


def pycolmap_device(pycolmap: Any) -> Any:
    runtime = pycolmap_feature_runtime()
    return pycolmap.Device.cuda if runtime["cudaValidated"] else pycolmap.Device.cpu
