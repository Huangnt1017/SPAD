"""将 PPTX 本地渲染为 PDF、逐页 PNG 和联系表，用于视觉优化闭环。

CLI:
    D:\\Anaconda3\\envs\\torchnew\\python.exe scripts\\render_ppt_preview.py \\
      --input model\\ppt\\SPAD_model_architectures_editable_combined.pptx \\
      --output-dir temp\\ppt_render\\manual --dpi 180

无参数运行：渲染当前合并版 PPT 到 ``temp/ppt_render/manual``。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "model" / "ppt" / "SPAD_model_architectures_editable_combined.pptx"
DEFAULT_OUTPUT = PROJECT_ROOT / "temp" / "ppt_render" / "manual"


@dataclass
class RenderConfig:
    input_path: Path
    output_dir: Path
    dpi: int = 180


def find_soffice() -> Path:
    """定位 LibreOffice soffice.exe。"""
    candidates = [
        Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = shutil.which("soffice")
    if found:
        return Path(found)
    raise FileNotFoundError("LibreOffice soffice.exe not found")


def find_pdftoppm() -> Path:
    """定位 Poppler pdftoppm.exe，包括 winget 安装目录。"""
    found = shutil.which("pdftoppm")
    if found:
        return Path(found)
    package_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    matches = sorted(package_root.glob("oschwartz10612.Poppler_*/poppler-*/Library/bin/pdftoppm.exe"))
    if matches:
        return matches[-1]
    raise FileNotFoundError("Poppler pdftoppm.exe not found")


def make_contact_sheet(png_paths: list[Path], output_path: Path) -> None:
    """把逐页 PNG 纵向拼成联系表。"""
    images = [Image.open(path).convert("RGB") for path in png_paths]
    if not images:
        raise RuntimeError("No rendered slide PNG files found")
    canvas_width = 1220
    row_height = 720
    sheet = Image.new("RGB", (canvas_width, row_height * len(images)), (230, 230, 230))
    for index, image in enumerate(images, 1):
        thumb = image.copy()
        thumb.thumbnail((1200, 675))
        row = Image.new("RGB", (canvas_width, row_height), "white")
        row.paste(thumb, ((canvas_width - thumb.width) // 2, 28))
        ImageDraw.Draw(row).text((15, 7), f"Slide {index}", fill="black")
        sheet.paste(row, (0, (index - 1) * row_height))
    sheet.save(output_path)


def render(config: RenderConfig) -> dict[str, object]:
    """执行 PPTX -> PDF -> PNG -> contact sheet。"""
    input_path = config.input_path.resolve()
    output_dir = config.output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if config.dpi <= 0:
        raise ValueError(f"dpi must be positive, got {config.dpi}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / f"{input_path.stem}.pdf"
    contact_path = output_dir / "contact_sheet.png"
    for path in [pdf_path, contact_path, *output_dir.glob("slide-*.png")]:
        if path.is_file():
            path.unlink()

    soffice = find_soffice()
    pdftoppm = find_pdftoppm()
    subprocess.run(
        [str(soffice), "--headless", "--convert-to", "pdf", "--outdir", str(output_dir), str(input_path)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    for _ in range(60):
        if pdf_path.is_file():
            break
        time.sleep(0.25)
    if not pdf_path.is_file():
        raise RuntimeError(f"LibreOffice did not produce PDF: {pdf_path}")

    subprocess.run(
        [str(pdftoppm), "-png", "-r", str(config.dpi), str(pdf_path), str(output_dir / "slide")],
        check=True,
        cwd=PROJECT_ROOT,
    )
    png_paths = sorted(output_dir.glob("slide-*.png"))
    make_contact_sheet(png_paths, contact_path)
    return {
        "pdf": str(pdf_path),
        "slides": [str(path) for path in png_paths],
        "contact_sheet": str(contact_path),
        "soffice": str(soffice),
        "pdftoppm": str(pdftoppm),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render PPTX to PDF and PNG previews")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input PPTX")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--dpi", type=int, default=180, help="PNG render DPI")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = render(RenderConfig(args.input, args.output_dir, args.dpi))
    for key, value in result.items():
        print(f"{key}={value}")
    return 0


def main_without_cli() -> None:
    result = render(RenderConfig(DEFAULT_INPUT, DEFAULT_OUTPUT, 180))
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        raise SystemExit(main())
    main_without_cli()
