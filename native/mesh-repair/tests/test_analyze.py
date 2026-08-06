from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


def write_ply(path: Path, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]) -> None:
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property double x",
        "property double y",
        "property double z",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines.extend(f"{x} {y} {z}" for x, y, z in vertices)
    lines.extend(f"3 {a} {b} {c}" for a, b, c in faces)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def cube() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices = [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 0, 4), (3, 4, 7),
    ]
    return vertices, faces


def run_analyze(executable: Path, workspace: Path, name: str,
                vertices: list[tuple[float, float, float]],
                faces: list[tuple[int, int, int]]) -> dict:
    input_path = workspace / f"{name}.ply"
    report_path = workspace / f"{name}.json"
    repeat_path = workspace / f"{name}-repeat.json"
    write_ply(input_path, vertices, faces)
    command = [str(executable), "analyze", "--input", str(input_path), "--report",
               str(report_path), "--voxel-size-m", "0.008"]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, f"{name}: {first.stderr}"
    command[command.index(str(report_path))] = str(repeat_path)
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 0, f"{name} repeat: {second.stderr}"
    first_bytes = report_path.read_bytes()
    second_bytes = repeat_path.read_bytes()
    assert first_bytes == second_bytes, f"{name}: analysis is not byte-for-byte deterministic"
    document = json.loads(first_bytes)
    assert document["schemaVersion"] == 1
    assert document["algorithmVersion"] == "1.0.0"
    assert document["status"] == "ok"
    assert document["inputMeshFingerprint"] == {
        "algorithm": "sha256",
        "value": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    assert document["mesh"]["vertexCount"] == len(vertices)
    assert document["mesh"]["triangleCount"] == len(faces)
    return document


def assert_close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def main() -> None:
    executable = Path(sys.argv[1]).resolve()
    workspace = Path(sys.argv[2]).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    version = subprocess.run([str(executable), "version", "--json"], capture_output=True,
                             text=True, check=False)
    assert version.returncode == 0, version.stderr
    version_document = json.loads(version.stdout)
    assert version_document["algorithmVersion"] == "1.0.0"
    assert version_document["backend"]["name"] == "CGAL"
    assert version_document["backend"]["version"].startswith("6.")

    cube_vertices, cube_faces = cube()
    closed = run_analyze(executable, workspace, "closed-cube", cube_vertices, cube_faces)
    assert closed["topology"] == {
        "boundaryLoopCount": 0,
        "connectedComponentCount": 1,
        "degenerateTriangleCount": 0,
        "duplicateTriangleCount": 0,
        "duplicateVertexCount": 0,
        "nonCycleBoundaryComponentCount": 0,
        "nonManifoldEdgeCount": 0,
        "nonManifoldVertexCount": 0,
        "selfIntersectionCount": 0,
    }

    missing = run_analyze(executable, workspace, "missing-face-cube", cube_vertices,
                          cube_faces[:2] + cube_faces[4:])
    assert missing["topology"]["boundaryLoopCount"] == 1
    loop = missing["boundaryLoops"][0]
    assert loop["vertexCount"] == 4
    assert_close(loop["perimeterM"], 4.0)
    assert_close(loop["approximateEnclosedAreaM2"], 1.0)
    assert_close(loop["diameterM"], math.sqrt(2.0))
    assert_close(loop["planeRmsResidualM"], 0.0)
    assert loop["loopId"].startswith("loop-")
    assert len(loop["orderedBoundaryPositions"]) == 4

    duplicate = run_analyze(executable, workspace, "duplicate-triangle", cube_vertices,
                            cube_faces + [cube_faces[0]])
    assert duplicate["topology"]["duplicateTriangleCount"] == 1

    degenerate = run_analyze(executable, workspace, "degenerate-triangle", cube_vertices,
                             cube_faces + [(0, 0, 1)])
    assert degenerate["topology"]["degenerateTriangleCount"] == 1

    bowtie_vertices = [(0, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)]
    bowtie_faces = [(0, 1, 2), (0, 3, 4)]
    bowtie = run_analyze(executable, workspace, "bowtie", bowtie_vertices, bowtie_faces)
    assert bowtie["topology"]["nonManifoldVertexCount"] == 1

    intersecting_vertices = [
        (-1, -1, 0), (1, -1, 0), (0, 1, 0),
        (0, -0.5, -1), (0, -0.5, 1), (0, 0.75, 0),
    ]
    intersecting = run_analyze(executable, workspace, "intersecting-triangles",
                               intersecting_vertices, [(0, 1, 2), (3, 4, 5)])
    assert intersecting["topology"]["selfIntersectionCount"] == 1

    disconnected_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
                             (10, 0, 0), (11, 0, 0), (10, 1, 0)]
    disconnected = run_analyze(executable, workspace, "disconnected-components",
                               disconnected_vertices, [(0, 1, 2), (3, 4, 5)])
    assert disconnected["topology"]["connectedComponentCount"] == 2
    assert len(disconnected["connectedComponents"]) == 2

    duplicate_vertex_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]
    duplicate_vertex = run_analyze(executable, workspace, "duplicate-vertex",
                                   duplicate_vertex_vertices, [(0, 1, 2)])
    assert duplicate_vertex["topology"]["duplicateVertexCount"] == 1

    invalid_input = workspace / "invalid.ply"
    invalid_report = workspace / "invalid.json"
    invalid_input.write_text("this is not a ply mesh\n", encoding="utf-8")
    invalid = subprocess.run(
        [str(executable), "analyze", "--input", str(invalid_input), "--report",
         str(invalid_report), "--voxel-size-m", "0.008"],
        capture_output=True, text=True, check=False,
    )
    assert invalid.returncode != 0
    assert "scanlan-mesh-repair:" in invalid.stderr
    invalid_document = json.loads(invalid_report.read_text(encoding="utf-8"))
    assert invalid_document["status"] == "error"
    assert invalid_document["error"]["code"] == "analysis_failed"
    assert invalid_document["error"]["message"]

    print("All mesh-repair analyzer fixtures passed.")


if __name__ == "__main__":
    main()
