"""
Assemble the anonymized reproducibility artifact promised in the manuscripts.

Two archives are produced:
  dp_audit_artifact.zip  - code, prompts, trained weights, dataset and held-out
                           specifications, and the per-cell records for the main
                           held-out comparison (2,160 records), the matched
                           disclosure test (1,800), the modality/prompting grid,
                           the Lorenz sweep, the two-seed replication, and the
                           hardware check.
  dp_audit_sweeps.zip    - summaries and configurations of the two large sweeps
                           (training-budget and balanced-seed studies); their
                           datasets and checkpoints regenerate from the recorded
                           seeds and configs.

Every text file in the staging tree is scanned against a banned-pattern list
(author names, usernames, e-mail addresses, local paths, credential shapes);
the build fails if any pattern matches. The .env file, git metadata, caches,
and run logs are never copied.

    python scripts/build_anonymized_artifact.py [--output DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAIN = [
    # code, prompts (bench/prompts.py), tests, configuration
    "bench", "scripts", "tests", "config.yaml", "requirements.txt",
    # dataset and held-out specifications
    "results/dataset",
    "results/heldout_llm_eval_set.json",
    "results/dataset_llm_sysid",
    "results/dataset_llm_sysid_context",
    # trained weights (primary models)
    "results/learned_models",
    # per-cell records: main grid, disclosure test, modality/prompting grid,
    # and the second temperature-0 run used for the repeatability analysis
    "results/llm_main",
    "results/llm_sysid",
    "results/checkpoints",
    "results/checkpoints_bon/run2",
    "results/hosted_repeat_plan",
    # aggregated records and analysis tables
    "results/summary_llm",
    "results/summary",
    "results/summary_boot",
    "results/summary_boot_hidden_equalized",
    "results/summary_boot_hidden_prior_changed_hidden",
    "results/two_seed_summary",
    "results/summary_round2",
    "results/summary_tsboot",
    "results/identifiability",
    # Lorenz-63 sweep (records, configs, weights, datasets)
    "results/lorenz_seed_sweep",
    # two-seed replication (summaries + weights; datasets regenerate from seeds)
    "results/seed_sweep_two_seed_focused/summary",
    "results/seed_sweep_two_seed_focused/checkpoints",
    # hardware check: tracked angles (Myers et al., Mendeley CC BY 4.0
    # listing; the deposit ships a GPLv3 license file, copied along)
    "results/realdata_myers/Video_Tracking_Data/Trial1",
    "results/realdata_myers/Video_Tracking_Data/Trial2",
    "results/realdata_myers/Video_Tracking_Data/Trial3",
    "results/realdata_myers/GNU_GPL_v3_License.txt",
]

SWEEPS = [
    "results/training_budget_study/summary",
    "results/training_budget_study/configs",
    "results/training_budget_study/run_config.json",
    "results/balanced_seed_study/summary",
    "results/balanced_seed_study/scientific_baselines_final/summary",
    "results/balanced_seed_study/dataset_inventory.json",
]

EXCLUDE_PARTS = {"__pycache__", ".git", ".venv"}
EXCLUDE_SUFFIX = {".log", ".pyc"}
# The builder excludes itself: its banned-pattern list would deanonymize.
EXCLUDE_NAMES = {".env", ".env.example", "build_anonymized_artifact.py",
                 ".DS_Store", "Thumbs.db", "desktop.ini"}

BANNED = re.compile(
    r"kandi|sriman|srinivasan|trishant|shahapure|shrithik|srima\b"
    r"|glorioussnk|leonschools|rickards|@gmail|@outlook"
    r"|[A-Za-z]:\\+Users\b|/home/\w+"
    r"|api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"
    r"|sk-[A-Za-z0-9]{20,}",
    re.IGNORECASE,
)
TEXT_SUFFIX = {".py", ".yaml", ".yml", ".txt", ".md", ".json", ".csv", ".cfg", ".toml"}

README = """\
# Auditing Learned Surrogates of Chaotic Dynamics - reproducibility artifact

Anonymized artifact for double-blind review. Layout:

  bench/, scripts/, tests/   evaluation harness, prompts (bench/prompts.py),
                             training / evaluation / analysis scripts
  config.yaml                model grid and run configuration (credentials are
                             read from an untracked .env and are not included)
  results/dataset/           nine training trajectories, one per (k, regime)
  results/heldout_llm_eval_set.json  shared held-out evaluation specification
  results/learned_models/    trained weights for the seven learned models
  results/llm_main/          per-cell records, main held-out comparison (2,160)
  results/llm_sysid/         per-cell records, matched disclosure test (1,800)
  results/checkpoints/       per-cell records, modality x prompting grid
  results/summary_llm/       aggregated tables used by the paper, including
                             realdata_myers.csv (hardware check) and
                             insample_training_trajectories.csv
  results/lorenz_seed_sweep/ Lorenz-63 sweep: configs, datasets, weights, records
  results/seed_sweep_two_seed_focused/  two-seed replication (summaries, weights)
  results/identifiability/   structural identifiability analysis outputs
  results/realdata_myers/    tracked angle series for the hardware check,
                             from the open dataset of Myers et al. (HardwareX
                             2020, Mendeley Data doi:10.17632/z4hvxjgtbz.2,
                             CC BY 4.0 listing; the deposit's GPLv3 license
                             file is included alongside)

Key reproduction commands (Python 3.13, packages in requirements.txt):

  python scripts/eval_realdata_myers.py        # hardware check table
  python scripts/analyze_realdata_bootstrap.py # hardware CIs (iid + block)
  python scripts/eval_insample.py              # in-sample vs held-out table
  python scripts/analyze_identifiability.py    # symmetry / identifiability
  python -m unittest discover -s tests         # harness tests

Hosted-LLM numbers are recomputed from the saved per-cell records; re-calling
an API is a new evaluation rather than a reproduction. All dataset and held-out
seeds are recorded in the configuration files. The training-budget and
balanced-seed sweeps are packaged separately in dp_audit_sweeps.zip.
"""


def excluded(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    return path.suffix in EXCLUDE_SUFFIX or path.name in EXCLUDE_NAMES


def copy_entry(rel: str, staging: Path) -> list[Path]:
    src = ROOT / rel
    copied = []
    if src.is_file():
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    elif src.is_dir():
        for f in sorted(src.rglob("*")):
            if f.is_file() and not excluded(f.relative_to(ROOT)):
                dst = staging / f.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                copied.append(dst)
    else:
        sys.exit(f"missing artifact input: {rel}")
    return copied


def scan(files: list[Path], staging: Path) -> None:
    hits = []
    for f in files:
        if f.suffix not in TEXT_SUFFIX:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in BANNED.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{f.relative_to(staging)}:{line}: {m.group(0)!r}")
    if hits:
        print("BANNED PATTERNS FOUND - build aborted:")
        for h in hits[:40]:
            print(" ", h)
        sys.exit(1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(name: str, entries: list[str], out_dir: Path) -> None:
    staging = out_dir / name
    if staging.exists():
        shutil.rmtree(staging)
    files: list[Path] = []
    for rel in entries:
        files.extend(copy_entry(rel, staging))
    readme = staging / "README.md"
    readme.write_text(README, encoding="utf-8")
    files.append(readme)
    scan(files, staging)
    with (staging / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for f in sorted(files):
            writer.writerow([f.relative_to(staging).as_posix(),
                             f.stat().st_size, sha256(f)])
    zip_path = out_dir / f"{name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging).as_posix())
    mb = zip_path.stat().st_size / 1e6
    print(f"{zip_path}  {len(files)} files  {mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT.parent / "artifact_build"))
    args = parser.parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    build("dp_audit_artifact", MAIN, out_dir)
    build("dp_audit_sweeps", SWEEPS, out_dir)


if __name__ == "__main__":
    main()
