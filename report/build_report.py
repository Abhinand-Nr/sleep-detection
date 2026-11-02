"""Build Sleep_Detection_Report.pdf from report/report.html.

    .venv/Scripts/python.exe report/make_figures.py   # only if figures need regenerating
    python report/build_report.py                      # prints the HTML to PDF with headless Edge/Chrome

Optional: if pypdfium2 is installed, page previews are written to report/preview/page-N.png
and the page count is printed.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "report" / "report.html"
PDF = ROOT / "Sleep_Detection_Report.pdf"

CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge", "google-chrome", "chromium", "chrome",
]


def find_browser():
    for c in CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("No Edge/Chrome found to print the PDF.")


def main():
    browser = find_browser()
    subprocess.run([
        browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        "--no-first-run", "--allow-file-access-from-files",
        f"--print-to-pdf={PDF}", HTML.as_uri(),
    ], check=True, capture_output=True)
    print("wrote", PDF)

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return
    doc = pdfium.PdfDocument(str(PDF))
    print("pages:", len(doc))
    out = ROOT / "report" / "preview"
    out.mkdir(exist_ok=True)
    for i, page in enumerate(doc):
        page.render(scale=2).to_pil().save(out / f"page-{i + 1}.png")


if __name__ == "__main__":
    main()
