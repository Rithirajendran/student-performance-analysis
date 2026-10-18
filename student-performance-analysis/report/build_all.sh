#!/usr/bin/env bash
# Two-pass build: pass 1 measures page numbers, pass 2 writes them into the contents lists. Then makes the PDF.
set -e
cd "$(dirname "$0")"
N=Student_Dropout_EDA_Report_RITHI_R_24AD091
python3 build_report.py >/dev/null
soffice --headless --convert-to pdf $N.docx >/dev/null 2>&1
python3 make_pages.py $N.pdf _toc_items.json _page_map.json
python3 build_report.py _page_map.json >/dev/null
soffice --headless --convert-to pdf $N.docx >/dev/null 2>&1
python3 make_pages.py $N.pdf _toc_items.json _page_map2.json
diff _page_map.json _page_map2.json && echo "page numbers stable"
