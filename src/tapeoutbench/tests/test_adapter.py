"""Tests for the TapeoutBench adapter, using a small fake upstream package."""

import tarfile
import tomllib
from pathlib import Path

import pytest

from tapeoutbench.adapter import TapeoutBenchAdapter, TapeoutBenchError, sha256_of

UPSTREAM_TOML = """\
schema_version = "1.0"

[task]
name = "simra-internal/{task_id}"
authors = [{{ name = "someone" }}]

[metadata]
difficulty = "hard"

[agent]
timeout_sec = 3600.0

[environment]
memory_mb = 8192
"""


def make_task(root: Path, task_id: str, *, with_solution: bool = True) -> None:
    task = root / task_id
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "task.toml").write_text(UPSTREAM_TOML.format(task_id=task_id))
    (task / "instruction.md").write_text(f"Brief for {task_id}\n")
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n")
    if with_solution:
        (task / "solution").mkdir()
        (task / "solution" / "solve.sh").write_text("true\n")


@pytest.fixture
def package(tmp_path: Path) -> Path:
    root = tmp_path / "tapeoutbench-v0.1" / "tasks"
    root.mkdir(parents=True)
    make_task(root, "pv1-drclvs-sg13g2")
    make_task(root, "ota1-2stage-sg13g2")
    return tmp_path / "tapeoutbench-v0.1"


@pytest.fixture
def archive(package: Path, tmp_path: Path) -> Path:
    path = tmp_path / "tapeoutbench-v0.1.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        tar.add(package, arcname=package.name)
    return path


def test_copies_tasks_and_rewrites_identity(package: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    written = TapeoutBenchAdapter(out, source=package).run()

    assert sorted(written) == ["ota1-2stage-sg13g2", "pv1-drclvs-sg13g2"]
    config = tomllib.loads((out / "pv1-drclvs-sg13g2" / "task.toml").read_text())
    assert config["task"]["name"] == "simra/tapeoutbench__pv1-drclvs-sg13g2"
    assert config["task"]["authors"] == [
        {"name": "Simra AI", "email": "info@simra.tech"}
    ]
    # Upstream settings are kept.
    assert config["agent"]["timeout_sec"] == 3600.0
    assert config["environment"]["memory_mb"] == 8192
    assert config["metadata"]["difficulty"] == "hard"
    assert config["metadata"]["source_task_id"] == "pv1-drclvs-sg13g2"
    # Other files are copied verbatim.
    assert (out / "pv1-drclvs-sg13g2" / "instruction.md").read_text() == (
        "Brief for pv1-drclvs-sg13g2\n"
    )


def test_archive_records_digest(archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    digest = sha256_of(archive)
    TapeoutBenchAdapter(out, source=archive, source_sha256=digest).run()
    config = tomllib.loads((out / "ota1-2stage-sg13g2" / "task.toml").read_text())
    assert config["metadata"]["source_package_sha256"] == digest


def test_archive_digest_mismatch_stops(archive: Path, tmp_path: Path) -> None:
    with pytest.raises(TapeoutBenchError, match="digest mismatch"):
        TapeoutBenchAdapter(
            tmp_path / "out", source=archive, source_sha256="0" * 64
        ).run()


def test_missing_solution_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "pkg"
    root.mkdir()
    make_task(root, "bgr1-pnp-sky130", with_solution=False)
    with pytest.raises(TapeoutBenchError, match="missing solution/solve.sh"):
        TapeoutBenchAdapter(tmp_path / "out", source=root).run()


def test_task_ids_limit_and_overwrite(package: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert TapeoutBenchAdapter(out, source=package, limit=1).run() == [
        "ota1-2stage-sg13g2"
    ]
    # Existing task is skipped without --overwrite, replaced with it.
    adapter = TapeoutBenchAdapter(out, source=package, task_ids=["ota1-2stage-sg13g2"])
    assert adapter.run() == []
    adapter = TapeoutBenchAdapter(
        out, source=package, task_ids=["ota1-2stage-sg13g2"], overwrite=True
    )
    assert adapter.run() == ["ota1-2stage-sg13g2"]
    with pytest.raises(TapeoutBenchError, match="Unknown task ids"):
        TapeoutBenchAdapter(out, source=package, task_ids=["nope"]).run()
