"""Build report.docx and report.pdf from REPORT.md.

Requires pandoc and Google Chrome (headless print-to-PDF).
    python docs/report/build_report.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = HERE / "REPORT.md"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

CSS = (HERE / "report.css").read_text()


def _image_tables_to_class(html: str) -> str:
    """Mark tables whose cells only contain images so CSS can lay them out as figure grids."""

    def mark(match: re.Match) -> str:
        table = match.group(0)
        text = re.sub(r"<[^>]+>", "", table).strip()
        return table.replace("<table>", '<table class="figs">', 1) if "<img" in table and not text else table

    return re.sub(r"<table>.*?</table>", mark, html, flags=re.S)


def build_pdf(tmp: Path) -> Path:
    html_path = tmp / "report.html"
    subprocess.run(
        [
            "pandoc",
            str(SRC),
            "-f",
            "gfm",
            "-t",
            "html5",
            "-s",
            "--metadata",
            "pagetitle=MLOps Assignment 1 Report",
            "--resource-path",
            str(HERE),
            "-o",
            str(html_path),
        ],
        check=True,
    )
    html = html_path.read_text()
    html = html.replace("</head>", f"<style>{CSS}</style></head>")
    html = _image_tables_to_class(html)
    # Resolve relative image paths against the report folder.
    html = re.sub(
        r'src="(\.\./[^"]+|[^":/"]+\.png)"', lambda m: f'src="{(HERE / m.group(1)).resolve().as_uri()}"', html
    )
    html_path.write_text(html)
    pdf = HERE / "report.pdf"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    return pdf


def build_docx(tmp: Path) -> Path:
    text = SRC.read_text()
    lines = []
    for line in text.splitlines():
        in_table = line.lstrip().startswith("|")
        width = "3.15in" if in_table else "6.5in"
        lines.append(IMG.sub(lambda m, w=width: f"![{m.group(1)}]({m.group(2)}){{width={w}}}", line))
    md = tmp / "report_docx.md"
    md.write_text("\n".join(lines))
    docx = HERE / "report.docx"
    subprocess.run(
        [
            "pandoc",
            str(md),
            "-f",
            "markdown+pipe_tables+link_attributes-yaml_metadata_block",
            "-t",
            "docx",
            "--resource-path",
            str(HERE),
            "-o",
            str(docx),
        ],
        check=True,
    )
    return docx


def main() -> None:
    tmp = HERE / ".build"
    tmp.mkdir(exist_ok=True)
    try:
        print("wrote", build_docx(tmp).relative_to(ROOT))
        print("wrote", build_pdf(tmp).relative_to(ROOT))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
