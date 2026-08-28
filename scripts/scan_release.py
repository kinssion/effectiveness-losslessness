#!/usr/bin/env python3
"""Scan the distributable tree for secrets, identities, paths, raw data, and weights."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SKIP_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".uv-cache",
    ".venv",
    "__pycache__",
}
TEXT_SUFFIXES = {
    "",
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
FORBIDDEN_BINARY_SUFFIXES = {".ckpt", ".mid", ".midi", ".pt", ".pth", ".safetensors"}
PATTERNS = {
    "windows_absolute_path": re.compile(
        r"[A-Za-z]:[\\/](?:Users|Documents|jobhunt|workspace)[\\/]", re.IGNORECASE
    ),
    "unix_private_path": re.compile(
        r"/(?:home/[^/\s]+|data/run[^/\s]*|mnt/[^/\s]+|scratch/[^/\s]+)"
    ),
    "email_or_login": re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "assigned_credential": re.compile(
        r"(?i)\b(?:api[_-]?key|password|passwd|client[_-]?secret|access[_-]?token)\b\s*[:=]\s*[\"'][^\"']{6,}[\"']"
    ),
    "ssh_endpoint": re.compile(r"(?i)\bssh\.[A-Za-z0-9.-]+|\bssh\s+[\w.-]+@"),
}


def included_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not any(part in SKIP_DIRECTORIES for part in path.relative_to(root).parts)
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
        and not path.name.endswith((".zip", ".tar", ".tar.gz"))
    ]


def scan(root: Path) -> tuple[list[str], int]:
    findings: list[str] = []
    scanned = 0
    for path in included_files(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in FORBIDDEN_BINARY_SUFFIXES:
            findings.append(f"forbidden_binary:{relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or path.name == "secret_scan_report.txt":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}:{relative}")
    return sorted(set(findings)), scanned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("secret_scan_report.txt"))
    args = parser.parse_args()
    root = args.root.resolve()
    findings, scanned = scan(root)
    status = "PASS" if not findings else "FAIL"
    lines = [
        "Effectiveness-Losslessness release scan",
        f"status: {status}",
        f"text_files_scanned: {scanned}",
        f"findings: {len(findings)}",
        "checks: secrets, private keys, login/email endpoints, private absolute paths, raw MIDI, model weights",
    ]
    lines.extend(f"finding: {finding}" for finding in findings)
    output = args.output if args.output.is_absolute() else root / args.output
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
