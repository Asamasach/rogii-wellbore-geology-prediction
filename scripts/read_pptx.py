"""Extract all text from the .pptx task description (stdlib only)."""
import re
import sys
import zipfile
from pathlib import Path

PPTX = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw\AI_wellbore_geology_prediction_task_en.pptx")
sys.stdout.reconfigure(encoding="utf-8")

# pptx = zip with ppt/slides/slide{N}.xml inside
TXT_RE = re.compile(r"<a:t[^>]*>([^<]*)</a:t>")
TAG_RE = re.compile(r"<[^>]+>")


def extract_slide_text(xml_bytes):
    s = xml_bytes.decode("utf-8", errors="replace")
    parts = TXT_RE.findall(s)
    return [p for p in parts if p.strip()]


with zipfile.ZipFile(PPTX) as z:
    slide_names = sorted(
        [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
        key=lambda n: int(re.search(r"slide(\d+)\.xml", n).group(1)),
    )
    print(f"# {PPTX.name} — {len(slide_names)} slides\n")
    for sn in slide_names:
        idx = int(re.search(r"slide(\d+)\.xml", sn).group(1))
        xml = z.read(sn)
        parts = extract_slide_text(xml)
        print(f"\n## SLIDE {idx}\n")
        for p in parts:
            print(p)
