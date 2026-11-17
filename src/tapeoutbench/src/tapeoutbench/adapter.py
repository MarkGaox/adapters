"""TapeoutBench -> Harbor adapter.

TapeoutBench v0.1 (Simra AI) is distributed by its authors as a package of tasks that
already follow the Harbor task layout (the authors ran every published trial on
Harbor 0.22.0). This adapter therefore does not rebuild tasks from raw data. It:

1. reads the upstream package (an unpacked directory or a .tar.gz archive, optionally
   checked against a pinned sha256),
2. copies each task directory unchanged,
3. rewrites only the registry identity in ``task.toml`` (``[task].name``,
   ``[task].authors``, ``[task].keywords``) from ``task-template/task.toml`` and records
   the package digest under ``[metadata]``, and
4. checks that every task still has the files Harbor needs.

Instructions, environments, verifiers and reference solutions are the authors' own.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tomlkit

TEMPLATE_DIR = Path(__file__).parent / "task-template"

# Files every upstream task must carry. A task may use a prebuilt image instead of a
# Dockerfile, so the environment check is done separately.
REQUIRED_FILES = ("task.toml", "instruction.md", "tests/test.sh", "solution/solve.sh")


class TapeoutBenchError(RuntimeError):
    """Raised when the upstream package is missing, altered, or malformed."""


@dataclass(frozen=True)
class UpstreamTask:
    task_id: str
    path: Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")


def _find_tasks_root(root: Path) -> Path:
    """Return the one directory whose children are the task directories.

    Accepts layouts such as ``<root>/<task>/``, ``<root>/tasks/<task>/`` or
    ``<root>/<package>/tasks/<task>/`` (a task is a directory with a task.toml).
    """
    parents = {p.parent.parent for p in root.rglob("task.toml") if p.parent != root}
    if not parents:
        raise TapeoutBenchError(
            f"No task directories (with task.toml) found under {root}"
        )
    if len(parents) > 1:
        found = ", ".join(sorted(str(p.relative_to(root)) for p in parents))
        raise TapeoutBenchError(f"Tasks found in more than one directory: {found}")
    return parents.pop()


def discover_tasks(tasks_root: Path) -> list[UpstreamTask]:
    tasks = [
        UpstreamTask(task_id=p.name, path=p)
        for p in sorted(tasks_root.iterdir())
        if p.is_dir() and (p / "task.toml").is_file()
    ]
    if not tasks:
        raise TapeoutBenchError(f"No tasks found in {tasks_root}")
    return tasks


def check_task(task: UpstreamTask) -> list[str]:
    """Return a list of problems with an upstream task directory (empty if fine)."""
    problems = [
        f"{task.task_id}: missing {rel}"
        for rel in REQUIRED_FILES
        if not (task.path / rel).is_file()
    ]
    config = tomlkit.parse((task.path / "task.toml").read_text())
    env_cfg = config.get("environment", {})
    has_dockerfile = (task.path / "environment" / "Dockerfile").is_file()
    has_compose = (task.path / "environment" / "docker-compose.yaml").is_file()
    if not (has_dockerfile or has_compose or env_cfg.get("docker_image")):
        problems.append(
            f"{task.task_id}: no environment/Dockerfile, docker-compose.yaml "
            "or [environment].docker_image"
        )
    return problems


def render_identity(task_id: str) -> tomlkit.TOMLDocument:
    """Load the identity overlay from the template with {task_id} filled in."""
    text = (TEMPLATE_DIR / "task.toml").read_text().replace("{task_id}", task_id)
    return tomlkit.parse(text)


def rewrite_task_toml(path: Path, task_id: str, package_sha256: str | None) -> None:
    """Overwrite the registry identity of a copied task, keeping everything else."""
    config = tomlkit.parse(path.read_text())
    overlay = render_identity(task_id)

    task_table = config.get("task")
    if task_table is None:
        task_table = tomlkit.table()
        config["task"] = task_table
    for key in ("name", "authors", "keywords"):
        task_table[key] = overlay["task"][key]

    metadata = config.get("metadata")
    if metadata is None:
        metadata = tomlkit.table()
        config["metadata"] = metadata
    metadata["source_benchmark"] = "TapeoutBench v0.1"
    metadata["source_task_id"] = task_id
    if package_sha256:
        metadata["source_package_sha256"] = package_sha256

    path.write_text(tomlkit.dumps(config))


class TapeoutBenchAdapter:
    def __init__(
        self,
        output_dir: Path,
        limit: int | None = None,
        overwrite: bool = False,
        task_ids: list[str] | None = None,
        source: Path | None = None,
        source_sha256: str | None = None,
        **kwargs,
    ):
        """
        Args:
            output_dir: Directory to write the Harbor tasks to.
            limit: Generate only the first N tasks (sorted by task id).
            overwrite: Replace task directories that already exist.
            task_ids: Only generate these upstream task ids.
            source: The upstream TapeoutBench package: an unpacked directory or a
                .tar.gz archive. The package is not public; request it from the
                authors (see README).
            source_sha256: Expected sha256 of the archive. Generation stops if the
                archive does not match, so a regenerated dataset is byte-identical.
        """
        self.output_dir = Path(output_dir)
        self.limit = limit
        self.overwrite = overwrite
        self.task_ids = task_ids
        self.source = Path(source) if source else None
        self.source_sha256 = source_sha256

    def _select(self, tasks: list[UpstreamTask]) -> list[UpstreamTask]:
        if self.task_ids:
            wanted = set(self.task_ids)
            unknown = wanted - {t.task_id for t in tasks}
            if unknown:
                raise TapeoutBenchError(f"Unknown task ids: {sorted(unknown)}")
            tasks = [t for t in tasks if t.task_id in wanted]
        if self.limit is not None:
            tasks = tasks[: max(0, self.limit)]
        return tasks

    def _generate_from(self, root: Path, package_sha256: str | None) -> list[str]:
        tasks = self._select(discover_tasks(_find_tasks_root(root)))

        problems = [p for task in tasks for p in check_task(task)]
        if problems:
            raise TapeoutBenchError(
                "Upstream package is incomplete:\n  " + "\n  ".join(problems)
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        for task in tasks:
            dest = self.output_dir / task.task_id
            if dest.exists():
                if not self.overwrite:
                    continue
                shutil.rmtree(dest)
            shutil.copytree(task.path, dest, symlinks=True)
            rewrite_task_toml(dest / "task.toml", task.task_id, package_sha256)
            written.append(task.task_id)
        return written

    def run(self) -> list[str]:
        if self.source is None:
            raise TapeoutBenchError(
                "--source is required: pass the TapeoutBench package "
                "(directory or .tar.gz) obtained from the authors."
            )
        if not self.source.exists():
            raise TapeoutBenchError(f"Source not found: {self.source}")

        if self.source.is_dir():
            if self.source_sha256:
                raise TapeoutBenchError(
                    "--source-sha256 applies to an archive; pass the .tar.gz instead "
                    "of an unpacked directory to pin the package."
                )
            return self._generate_from(self.source, None)

        actual = sha256_of(self.source)
        if self.source_sha256 and actual != self.source_sha256.lower():
            raise TapeoutBenchError(
                f"Package digest mismatch: expected {self.source_sha256}, got {actual}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            _safe_extract(self.source, Path(tmp))
            return self._generate_from(Path(tmp), actual)
