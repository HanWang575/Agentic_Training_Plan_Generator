"""Run Marker on one PDF with GPU requested and OCR toggled.

This script is for manual parser-quality testing. It runs Marker twice by
default:

1. OCR off: normal Marker parsing, with no `--force_ocr`.
2. OCR on: Marker parsing with `--force_ocr`.

Each run writes its own output folder and the script creates a small JSON
summary so you can compare runtime, GPU availability, Markdown output size, and
table-looking blocks.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from pathlib import Path
from shutil import which


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace with PDF path, output directory, device, mode, and
        timeout settings.
    """

    parser = argparse.ArgumentParser(description="Test Marker GPU parsing with OCR on/off.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF to parse.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tmp") / "marker_gpu_ocr_tests",
        help="Folder where Marker outputs and summary files will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Value for TORCH_DEVICE. Use cuda for NVIDIA GPU, cpu for CPU fallback.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "ocr_off", "ocr_on"],
        default="both",
        help="Which Marker run to execute.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Maximum seconds allowed for each Marker run.",
    )
    parser.add_argument(
        "--output-format",
        default="markdown",
        choices=["markdown", "json", "html"],
        help="Marker output format to request.",
    )
    parser.add_argument(
        "--progress-seconds",
        type=int,
        default=15,
        help="Seconds between heartbeat progress messages while Marker is running.",
    )
    parser.add_argument(
        "--page-batch-size",
        type=int,
        default=20,
        help=(
            "Parse pages in batches so the script can show an overall page progress bar. "
            "Set to 0 to run Marker once for the whole PDF and show only Marker logs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Run Marker conversion jobs and write a comparison summary.

    Returns:
        int: process exit code. Zero means all requested runs completed.
    """

    args = parse_args()
    pdf_path = args.pdf_path.resolve()
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    marker_path = find_marker_single()
    if marker_path is None:
        print("Could not find marker_single. Is marker-pdf installed in this venv?", file=sys.stderr)
        return 2

    output_root = (args.output_dir / pdf_path.stem).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    modes = ["ocr_off", "ocr_on"] if args.mode == "both" else [args.mode]
    page_count = count_pdf_pages(pdf_path)

    summary = {
        "pdf_path": str(pdf_path),
        "page_count": page_count,
        "marker_single": str(marker_path),
        "requested_device": args.device,
        "torch": torch_report(),
        "runs": [],
    }
    print(json.dumps({"torch": summary["torch"]}, indent=2))
    if args.device.startswith("cuda") and not summary["torch"].get("cuda_available"):
        print(
            "\nCUDA was requested, but this Python environment's Torch cannot use CUDA. "
            "Install a CUDA-enabled torch build in the same venv, or rerun with --device cpu.",
            file=sys.stderr,
        )
        summary_path = output_root / "marker_run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Summary written to: {summary_path}")
        return 2

    all_ok = True
    for mode in modes:
        run_summary = run_marker(
            marker_path=marker_path,
            pdf_path=pdf_path,
            output_dir=output_root / mode,
            output_format=args.output_format,
            device=args.device,
            force_ocr=(mode == "ocr_on"),
            timeout_seconds=args.timeout_seconds,
            progress_seconds=args.progress_seconds,
            page_count=page_count,
            page_batch_size=args.page_batch_size,
        )
        summary["runs"].append(run_summary)
        all_ok = all_ok and run_summary["returncode"] == 0
        print(json.dumps(run_summary, indent=2, ensure_ascii=False))

    summary_path = output_root / "marker_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary written to: {summary_path}")
    return 0 if all_ok else 1


def find_marker_single() -> Path | None:
    """Find Marker's `marker_single` executable.

    Returns:
        Path or None. Checks the active venv Scripts folder before PATH so
        global Python entry points do not accidentally run with the wrong Torch.
    """

    scripts_dir = Path(sys.executable).resolve().parent
    for name in ("marker_single.exe", "marker_single"):
        candidate = scripts_dir / name
        if candidate.exists():
            return candidate

    path_from_env = which("marker_single") or which("marker_single.exe")
    if path_from_env:
        return Path(path_from_env).resolve()
    return None


def torch_report() -> dict:
    """Return a small GPU availability report from PyTorch.

    Returns:
        dict: torch import status, CUDA availability, device count, and names.
    """

    try:
        import torch
    except Exception as exc:
        return {"import_ok": False, "error": str(exc)}

    report = {
        "import_ok": True,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [],
    }
    if torch.cuda.is_available():
        report["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return report


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF.

    Parameters:
        pdf_path: path to a PDF file.
    Returns:
        int: page count, or 0 if the count cannot be read.
    """

    try:
        import fitz

        with fitz.open(pdf_path) as document:
            return int(document.page_count)
    except Exception:
        return 0


def run_marker(
    marker_path: Path,
    pdf_path: Path,
    output_dir: Path,
    output_format: str,
    device: str,
    force_ocr: bool,
    timeout_seconds: int,
    progress_seconds: int,
    page_count: int,
    page_batch_size: int,
) -> dict:
    """Run one Marker conversion.

    Parameters:
        marker_path: path to `marker_single`.
        pdf_path: PDF file path.
        output_dir: output folder for this run.
        output_format: Marker output format, usually `markdown`.
        device: TORCH_DEVICE value, usually `cuda`.
        force_ocr: if true, pass `--force_ocr`.
        timeout_seconds: maximum runtime for this run.
        progress_seconds: heartbeat interval while the process is running.
        page_count: total PDF page count.
        page_batch_size: pages per Marker subprocess. Use 0 for one whole-file run.
    Returns:
        dict: run status, timings, output files, and table summary.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "ocr_on" if force_ocr else "ocr_off"
    if page_batch_size > 0 and page_count > 0:
        return run_marker_in_page_batches(
            marker_path=marker_path,
            pdf_path=pdf_path,
            output_dir=output_dir,
            output_format=output_format,
            device=device,
            force_ocr=force_ocr,
            timeout_seconds=timeout_seconds,
            progress_seconds=progress_seconds,
            page_count=page_count,
            page_batch_size=page_batch_size,
            mode=mode,
        )

    command = [
        str(marker_path),
        str(pdf_path),
        "--output_format",
        output_format,
        "--output_dir",
        str(output_dir),
    ]
    if force_ocr:
        command.append("--force_ocr")

    env = os.environ.copy()
    env["TORCH_DEVICE"] = device

    print(f"\n=== Marker run started: {mode} ===")
    print(f"Output dir: {output_dir}")
    print(f"Command: {' '.join(command)}")
    print(f"TORCH_DEVICE={device}")

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(pdf_path.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_queue: Queue[tuple[str, str | None]] = Queue()
    stdout_thread = threading.Thread(
        target=_read_stream,
        args=("stdout", process.stdout, output_queue),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream,
        args=("stderr", process.stderr, output_queue),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    seen_files: set[Path] = set()
    next_heartbeat = started + max(1, progress_seconds)
    timed_out = False

    while True:
        now = time.perf_counter()
        if now - started > timeout_seconds:
            timed_out = True
            process.kill()
            print(f"[{mode}] timed out after {timeout_seconds} seconds; process killed.")
            break

        try:
            stream_name, line = output_queue.get(timeout=0.5)
            if line is None:
                pass
            elif stream_name == "stdout":
                stdout_lines.append(line)
                print(f"[{mode} stdout] {line}", flush=True)
            else:
                stderr_lines.append(line)
                print(f"[{mode} stderr] {line}", flush=True)
        except Empty:
            pass

        if now >= next_heartbeat:
            elapsed = round(now - started, 1)
            new_files = _new_output_files(output_dir, seen_files)
            file_note = f"; new files: {', '.join(path.name for path in new_files[:5])}" if new_files else ""
            print(f"[{mode} progress] elapsed {elapsed}s; still running{file_note}", flush=True)
            next_heartbeat = now + max(1, progress_seconds)

        if process.poll() is not None:
            break

    returncode = None if timed_out else process.wait()
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    _drain_queue(output_queue, stdout_lines, stderr_lines, mode)

    elapsed_seconds = round(time.perf_counter() - started, 2)
    stdout = "\n".join(stdout_lines)
    stderr = "\n".join(stderr_lines)
    stdout_path = output_dir / "marker_stdout.txt"
    stderr_path = output_dir / "marker_stderr.txt"
    stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(stderr, encoding="utf-8", errors="replace")

    output_files = [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()]
    markdown_files = sorted(output_dir.rglob("*.md"))
    markdown_summary = summarize_markdown(markdown_files[0]) if markdown_files else {}

    return {
        "mode": mode,
        "force_ocr": force_ocr,
        "torch_device_env": device,
        "command": command,
        "timed_out": timed_out,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_dir": str(output_dir),
        "output_files": output_files,
        "markdown_summary": markdown_summary,
    }


def run_marker_in_page_batches(
    marker_path: Path,
    pdf_path: Path,
    output_dir: Path,
    output_format: str,
    device: str,
    force_ocr: bool,
    timeout_seconds: int,
    progress_seconds: int,
    page_count: int,
    page_batch_size: int,
    mode: str,
) -> dict:
    """Run Marker over page ranges and display an overall page progress bar."""

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    batches = list(page_ranges(page_count, page_batch_size))
    batch_summaries = []
    completed_pages = 0
    all_ok = True
    print(f"\n=== Marker batched run started: {mode} ===")
    print(f"Pages: {page_count}; batch size: {page_batch_size}; batches: {len(batches)}")
    render_page_progress(mode, completed_pages, page_count, started)

    for batch_index, (start_page, end_page) in enumerate(batches, start=1):
        batch_dir = output_dir / f"pages_{start_page + 1:04d}_{end_page + 1:04d}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{mode}] batch {batch_index}/{len(batches)}: "
            f"pages {start_page + 1}-{end_page + 1}"
        )
        summary = run_marker_single_process(
            marker_path=marker_path,
            pdf_path=pdf_path,
            output_dir=batch_dir,
            output_format=output_format,
            device=device,
            force_ocr=force_ocr,
            timeout_seconds=timeout_seconds,
            progress_seconds=progress_seconds,
            mode=mode,
            page_range=f"{start_page}-{end_page}",
        )
        batch_summaries.append(summary)
        all_ok = all_ok and summary["returncode"] == 0
        if summary["returncode"] != 0:
            print(f"[{mode}] stopping after failed batch {batch_index}.")
            break
        completed_pages = end_page + 1
        render_page_progress(mode, completed_pages, page_count, started)

    elapsed_seconds = round(time.perf_counter() - started, 2)
    output_files = [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()]
    markdown_files = sorted(output_dir.rglob("*.md"))
    markdown_summary = summarize_many_markdown(markdown_files)
    return {
        "mode": mode,
        "force_ocr": force_ocr,
        "torch_device_env": device,
        "batched_by_page": True,
        "page_count": page_count,
        "page_batch_size": page_batch_size,
        "completed_pages": completed_pages,
        "batch_count": len(batches),
        "timed_out": any(item["timed_out"] for item in batch_summaries),
        "returncode": 0 if all_ok else 1,
        "elapsed_seconds": elapsed_seconds,
        "output_dir": str(output_dir),
        "output_files": output_files,
        "batch_summaries": batch_summaries,
        "markdown_summary": markdown_summary,
    }


def run_marker_single_process(
    marker_path: Path,
    pdf_path: Path,
    output_dir: Path,
    output_format: str,
    device: str,
    force_ocr: bool,
    timeout_seconds: int,
    progress_seconds: int,
    mode: str,
    page_range: str | None = None,
) -> dict:
    """Run one Marker subprocess and stream its logs."""

    command = [
        str(marker_path),
        str(pdf_path),
        "--output_format",
        output_format,
        "--output_dir",
        str(output_dir),
    ]
    if page_range:
        command.extend(["--page_range", page_range])
    if force_ocr:
        command.append("--force_ocr")

    env = os.environ.copy()
    env["TORCH_DEVICE"] = device
    print(f"Command: {' '.join(command)}")
    print(f"TORCH_DEVICE={device}")

    return _run_process_with_live_logs(
        command=command,
        cwd=pdf_path.parent,
        env=env,
        output_dir=output_dir,
        mode=mode,
        timeout_seconds=timeout_seconds,
        progress_seconds=progress_seconds,
    )


def _run_process_with_live_logs(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    mode: str,
    timeout_seconds: int,
    progress_seconds: int,
) -> dict:
    """Run a subprocess, stream its logs, and save stdout/stderr files."""

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    output_queue: Queue[tuple[str, str | None]] = Queue()
    stdout_thread = threading.Thread(
        target=_read_stream,
        args=("stdout", process.stdout, output_queue),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream,
        args=("stderr", process.stderr, output_queue),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    seen_files: set[Path] = set()
    next_heartbeat = started + max(1, progress_seconds)
    timed_out = False

    while True:
        now = time.perf_counter()
        if now - started > timeout_seconds:
            timed_out = True
            process.kill()
            print(f"[{mode}] timed out after {timeout_seconds} seconds; process killed.")
            break

        try:
            stream_name, line = output_queue.get(timeout=0.5)
            if line is None:
                pass
            elif stream_name == "stdout":
                stdout_lines.append(line)
                print(f"[{mode} stdout] {line}", flush=True)
            else:
                stderr_lines.append(line)
                print(f"[{mode} stderr] {line}", flush=True)
        except Empty:
            pass

        if now >= next_heartbeat:
            elapsed = round(now - started, 1)
            new_files = _new_output_files(output_dir, seen_files)
            file_note = f"; new files: {', '.join(path.name for path in new_files[:5])}" if new_files else ""
            print(f"[{mode} progress] elapsed {elapsed}s; current batch still running{file_note}", flush=True)
            next_heartbeat = now + max(1, progress_seconds)

        if process.poll() is not None:
            break

    returncode = None if timed_out else process.wait()
    stdout_thread.join(timeout=1)
    stderr_thread.join(timeout=1)
    _drain_queue(output_queue, stdout_lines, stderr_lines, mode)

    elapsed_seconds = round(time.perf_counter() - started, 2)
    stdout_path = output_dir / "marker_stdout.txt"
    stderr_path = output_dir / "marker_stderr.txt"
    stdout_path.write_text("\n".join(stdout_lines), encoding="utf-8", errors="replace")
    stderr_path.write_text("\n".join(stderr_lines), encoding="utf-8", errors="replace")

    markdown_files = sorted(output_dir.rglob("*.md"))
    markdown_summary = summarize_markdown(markdown_files[0]) if markdown_files else {}
    return {
        "command": command,
        "timed_out": timed_out,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "output_dir": str(output_dir),
        "output_files": [str(path) for path in sorted(output_dir.rglob("*")) if path.is_file()],
        "markdown_summary": markdown_summary,
    }


def page_ranges(page_count: int, page_batch_size: int) -> list[tuple[int, int]]:
    """Return inclusive zero-based page ranges."""

    ranges = []
    start = 0
    while start < page_count:
        end = min(page_count - 1, start + page_batch_size - 1)
        ranges.append((start, end))
        start = end + 1
    return ranges


def render_page_progress(mode: str, completed_pages: int, page_count: int, started: float) -> None:
    """Print a page progress bar with elapsed time and ETA."""

    width = 30
    ratio = completed_pages / page_count if page_count else 0
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - started
    pages_per_second = completed_pages / elapsed if elapsed > 0 and completed_pages else 0
    remaining_pages = max(0, page_count - completed_pages)
    eta_seconds = int(remaining_pages / pages_per_second) if pages_per_second else None
    eta_text = format_duration(eta_seconds) if eta_seconds is not None else "estimating"
    percent = ratio * 100
    print(
        f"[{mode} overall] |{bar}| {completed_pages}/{page_count} pages "
        f"({percent:5.1f}%) elapsed {format_duration(int(elapsed))} ETA {eta_text}",
        flush=True,
    )


def format_duration(seconds: int) -> str:
    """Format seconds as h/m/s text."""

    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _read_stream(stream_name: str, stream, output_queue: Queue[tuple[str, str | None]]) -> None:
    """Read subprocess output lines into a queue for live progress display."""

    try:
        for line in iter(stream.readline, ""):
            output_queue.put((stream_name, line.rstrip()))
    finally:
        output_queue.put((stream_name, None))


def _drain_queue(
    output_queue: Queue[tuple[str, str | None]],
    stdout_lines: list[str],
    stderr_lines: list[str],
    mode: str,
) -> None:
    """Print and store any subprocess output that arrived near shutdown."""

    while True:
        try:
            stream_name, line = output_queue.get_nowait()
        except Empty:
            return
        if line is None:
            continue
        if stream_name == "stdout":
            stdout_lines.append(line)
            print(f"[{mode} stdout] {line}", flush=True)
        else:
            stderr_lines.append(line)
            print(f"[{mode} stderr] {line}", flush=True)


def _new_output_files(output_dir: Path, seen_files: set[Path]) -> list[Path]:
    """Return output files that appeared since the previous heartbeat."""

    current_files = {path for path in output_dir.rglob("*") if path.is_file()}
    new_files = sorted(current_files - seen_files)
    seen_files.update(current_files)
    return new_files


def summarize_markdown(markdown_path: Path) -> dict:
    """Summarize Marker Markdown output for quick table inspection.

    Parameters:
        markdown_path: Markdown output file.
    Returns:
        dict: character count, line count, table count, and short table samples.
    """

    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    tables = extract_markdown_tables(text)
    return {
        "markdown_path": str(markdown_path),
        "characters": len(text),
        "lines": text.count("\n") + 1,
        "table_count": len(tables),
        "table_samples": tables[:5],
    }


def summarize_many_markdown(markdown_paths: list[Path]) -> dict:
    """Summarize multiple Markdown files from a batched Marker run."""

    total_characters = 0
    total_lines = 0
    table_samples = []
    for markdown_path in markdown_paths:
        text = markdown_path.read_text(encoding="utf-8", errors="replace")
        total_characters += len(text)
        total_lines += text.count("\n") + 1
        table_samples.extend(extract_markdown_tables(text))
    return {
        "markdown_file_count": len(markdown_paths),
        "characters": total_characters,
        "lines": total_lines,
        "table_count": len(table_samples),
        "table_samples": table_samples[:5],
    }


def extract_markdown_tables(text: str) -> list[str]:
    """Extract Markdown table-looking blocks.

    Parameters:
        text: Markdown text.
    Returns:
        list[str]: table samples, each truncated for console readability.
    """

    lines = text.splitlines()
    tables = []
    index = 0
    separator_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
    while index < len(lines) - 1:
        if "|" in lines[index] and separator_re.match(lines[index + 1]):
            block = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index].strip():
                block.append(lines[index])
                index += 1
            sample = "\n".join(block)
            tables.append(sample[:1200])
            continue
        index += 1
    return tables


if __name__ == "__main__":
    raise SystemExit(main())
