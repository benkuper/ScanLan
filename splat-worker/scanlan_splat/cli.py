from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .train import train_dataset


def _cuda_smoke_test() -> None:
    import torch
    from gsplat.rendering import rasterization_2dgs

    means = torch.tensor([[0.0, 0.0, 2.0]], device="cuda", requires_grad=True)
    quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    scales = torch.full((1, 3), 0.05, device="cuda")
    opacities = torch.full((1,), 0.8, device="cuda")
    colors = torch.tensor([[0.3, 0.6, 0.9]], device="cuda")
    view = torch.eye(4, device="cuda").unsqueeze(0)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 3.5], [0.0, 8.0, 3.5], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    rendering, _, normals, _surface_normals, distortion, _median, _ = rasterization_2dgs(
        means=means,
        quats=quaternions,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=view,
        Ks=intrinsics,
        width=8,
        height=8,
        packed=True,
        render_mode="RGB+ED",
        distloss=True,
    )
    (rendering.sum() + normals.sum() + distortion.sum()).backward()
    torch.cuda.synchronize()
    if means.grad is None or not torch.isfinite(means.grad).all():
        raise RuntimeError("gsplat CUDA backward smoke test produced invalid gradients")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="scanlan-splat")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--project", type=Path, required=True)
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=30_000)
    train.add_argument("--resume", action="store_true")
    diagnostics = commands.add_parser("diagnostics")
    diagnostics.add_argument("--require-cuda", action="store_true")
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "diagnostics":
            import torch
            import gsplat

            cuda_available = torch.cuda.is_available()
            if arguments.require_cuda and cuda_available:
                _cuda_smoke_test()
            print(json.dumps({"version": __version__, "cuda": cuda_available, "cudaSmokeTest": cuda_available and arguments.require_cuda, "device": torch.cuda.get_device_name(0) if cuda_available else None, "torch": torch.__version__, "gsplat": getattr(gsplat, "__version__", "unknown")}))
            return 2 if arguments.require_cuda and not cuda_available else 0
        result = train_dataset(
            arguments.dataset.resolve(),
            arguments.project.resolve(),
            max(1_000, min(arguments.iterations, 100_000)),
            arguments.resume,
        )
        print(json.dumps(result))
        return 0
    except Exception as error:
        print(f"scanlan-splat: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
