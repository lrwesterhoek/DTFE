"""Assemble report/index.html: head + body (with {{name}} placeholders) + generated tables; copy figures.
   build_report.py <cmpdir>"""
import re
import shutil
import sys
from pathlib import Path

C = Path(sys.argv[1])
R = C / "report"
(R / "fig").mkdir(parents=True, exist_ok=True)
tables = {}
cur = None
for line in (C / "fig" / "tables.html").read_text().splitlines():
    m = re.match(r"<!-- (\w+) -->", line)
    if m:
        cur = m.group(1)
        tables[cur] = ""
    elif cur:
        tables[cur] += line + "\n"
tables["features"] = (C / "report_features.html").read_text()
body = (C / "report_body.html").read_text()
missing = [k for k in re.findall(r"\{\{(\w+)\}\}", body) if k not in tables]
if missing:
    sys.exit(f"missing tables: {missing}")
body = re.sub(r"\{\{(\w+)\}\}", lambda m: f'<div class="tablewrap">{tables[m.group(1)]}</div>'
              if not tables[m.group(1)].lstrip().startswith('<div class="tablewrap">') else tables[m.group(1)], body)
used = set(re.findall(r'src="fig/([\w.\-]+)"', body))
for name in used:
    shutil.copy2(C / "fig" / name, R / "fig" / name)
(R / "index.html").write_text((C / "report_head.html").read_text() + body)
print("report:", R / "index.html", "figures:", sorted(used))
