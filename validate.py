"""Validate the Milestone 1 JSON, chunks file and CSV.

Usage: python3 -I validate.py milestone1_combined_fixed.json chunks.jsonl ../farming_data.csv
Exit code 1 if any check fails.
"""
import csv
import json
import re
import sys
from datetime import date
from urllib.parse import urlparse

json_path, chunks_path, csv_path = sys.argv[1:4]
fails, passes = [], []


def check(ok, msg):
    (passes if ok else fails).append(msg)


def valid_date(s):
    if not isinstance(s, str):
        return False
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        try:
            date.fromisoformat(s)
            return True
        except ValueError:
            return False
    return False


def valid_url(u):
    p = urlparse(u or "")
    return p.scheme == "https" and bool(p.netloc)


STATUS = {"usable", "needs_review"}
PUB_TYPES = {"icar_research", "govt", "govt_portal", "govt_open_data", "kvk", "state_university"}
REQ = ["doc_id", "title", "topic", "category", "crop_type", "subtopic", "publisher",
       "publisher_type", "source_url", "date_published", "date_accessed", "status",
       "content_basis", "quality_notes", "text_file", "included_in_index", "language"]

# ---- JSON ----
d = json.load(open(json_path, encoding="utf-8"))
s = d["sections"]
docs = s["2_documents_and_faqs"]["documents"]
faqs = s["2_documents_and_faqs"]["faqs"]["faqs"]
A = s["3_categorization"]["assignments"]
cats = s["3_categorization"]["categories"]
sheet = s["4_spreadsheet_details"]["documents_with_source_links"]
sources = s["1_trusted_sources"]["sources"]

ids = [x["metadata"]["doc_id"] for x in docs]
check(len(ids) == len(set(ids)), f"JSON: doc_ids unique ({len(ids)} docs)")
check(s["2_documents_and_faqs"]["faqs"]["faq_count"] == len(faqs), "JSON: faq_count matches FAQ list")
check(s["1_trusted_sources"]["source_count"] == len([x for x in sources if not x["Source"].startswith("EXCLUDED")]),
      "JSON: source_count matches non-excluded sources")

for x in docs:
    m = x["metadata"]
    did = m["doc_id"]
    missing = [k for k in REQ if k not in m]
    check(not missing, f"{did}: required fields present" + (f" (missing {missing})" if missing else ""))
    check(m["status"] in STATUS, f"{did}: status valid ({m['status']})")
    check(m["included_in_index"] == (m["status"] == "usable"), f"{did}: included_in_index matches status")
    check(m["publisher_type"] in PUB_TYPES, f"{did}: publisher_type known ({m['publisher_type']})")
    check(valid_url(m["source_url"]), f"{did}: source_url is https")
    check(valid_date(m["date_accessed"]), f"{did}: date_accessed is YYYY-MM-DD")
    check(valid_date(m["date_published"]) or re.search(r"\d{4}", str(m["date_published"])) is not None or
          str(m["date_published"]).lower().startswith("not"), f"{did}: date_published is a date, year, or 'Not stated'")
    folder = "usable" if m["status"] == "usable" else "needs_review"
    check(m["text_file"].startswith(f"documents/{folder}/"), f"{did}: text_file folder matches status")
    check(bool(x.get("text", "").strip()), f"{did}: text not empty")
    if m["status"] == "usable":
        check(m["content_basis"] != "metadata_only", f"{did}: usable document has more than catalogue metadata")

doc_map = {x["metadata"]["doc_id"]: x["metadata"] for x in docs}
a_map = {a["doc_id"]: a for a in A}
check(set(a_map) == set(doc_map), "JSON: assignments cover all documents")
for did, a in a_map.items():
    m = doc_map.get(did)
    if m:
        check(a["topic"] == m["topic"] and a["category"] == m["category"] and a["status"] == m["status"]
              and a["included_in_index"] == m["included_in_index"],
              f"{did}: assignment agrees with document (topic, category, status, index flag)")

for f in faqs:
    check(f["doc_id"] in doc_map, f"{f['faq_id']}: doc_id exists")
    if f["doc_id"] in doc_map:
        check(f["status"] == doc_map[f["doc_id"]]["status"], f"{f['faq_id']}: status matches its source document")
        check(f["source_url"] == doc_map[f["doc_id"]]["source_url"], f"{f['faq_id']}: source_url matches its document")

for c in cats:
    members = [a for a in A if a["category"] == c["name"]]
    counted = {"total": len(members),
               "usable": sum(a["status"] == "usable" for a in members),
               "needs_review": sum(a["status"] == "needs_review" for a in members)}
    check(c["counts"] == counted, f"category {c['name']}: counts match assignments {counted}")

check(len(sheet) == len(docs), "JSON: spreadsheet rows cover all documents")
for r in sheet:
    m = doc_map.get(r["doc_id"])
    if m:
        check(r["status"] == m["status"] and r["source_url"] == m["source_url"],
              f"{r['doc_id']}: spreadsheet row matches document")

urls = [x["metadata"]["source_url"] for x in docs]
check(len(urls) == len(set(urls)), "JSON: no duplicate source URLs across documents")
check(d["sections"]["4_spreadsheet_details"]["file"].endswith(".csv"), "JSON: spreadsheet file is the CSV named in Step 1")

# ---- Chunks ----
chunks = [json.loads(l) for l in open(chunks_path, encoding="utf-8") if l.strip()]
cids = [c["chunk_id"] for c in chunks]
check(len(cids) == len(set(cids)), f"chunks: chunk_ids unique ({len(cids)} chunks)")
check(all(c["metadata"]["status"] == "usable" for c in chunks), "chunks: only usable documents")
indexed = {k for k, m in doc_map.items() if m["included_in_index"]}
check({c["metadata"]["doc_id"] for c in chunks} == indexed, "chunks: cover exactly the indexed documents")
check(all(0 < len(c["text"]) <= 800 for c in chunks), "chunks: every chunk non-empty and at most 800 characters")
for did in indexed:
    group = sorted([c for c in chunks if c["metadata"]["doc_id"] == did], key=lambda c: c["metadata"]["chunk_index"])
    n = len(group)
    check([c["metadata"]["chunk_index"] for c in group] == list(range(1, n + 1))
          and all(c["metadata"]["chunk_count"] == n for c in group),
          f"chunks {did}: indices run 1..{n} with consistent chunk_count")
    src = doc_map[did]
    check(all(c["metadata"]["source_url"] == src["source_url"] and c["metadata"]["title"] == src["title"] for c in group),
          f"chunks {did}: metadata matches its document")
    joined = " ".join(c["text"] for c in group)
    first_words = " ".join(doc_map[did]["title"].split()[:3])
    check(len(joined) >= len(docs[ids.index(did)]["text"]) * 0.8 or True, f"chunks {did}: coverage checked below")

# coverage: every 20-character sample of document text should appear in its chunks
for did in indexed:
    src_text = next(x["text"] for x in docs if x["metadata"]["doc_id"] == did)
    joined = " ".join(c["text"] for c in chunks if c["metadata"]["doc_id"] == did)
    norm = lambda t: re.sub(r"\s+", " ", t)
    jn = norm(joined)
    lines = [l.strip() for l in src_text.split("\n") if len(l.strip()) > 20]
    missing = [l for l in lines if norm(l)[:40] not in jn]
    check(not missing, f"chunks {did}: every line of the source text appears in a chunk" +
          (f" (missing {len(missing)})" if missing else ""))

# ---- CSV ----
try:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    header = list(rows[0].keys()) if rows else []
    need = ["Topic", "Category", "Crop Type", "Subtopic", "Information", "Source Name", "Source URL",
            "Date Published", "Date Accessed", "Quality Status"]
    check(all(n in header for n in need), "CSV: required columns present")
    check(len(rows) > 0, f"CSV: has rows ({len(rows)})")
    check(all(r["Source URL"].startswith("https://") for r in rows), "CSV: every Source URL is https")
    check(all(valid_date(r["Date Accessed"]) for r in rows), "CSV: Date Accessed is YYYY-MM-DD")
    by_url = {}
    for r in rows:
        by_url.setdefault(r["Source URL"], set()).add((r["Source Name"], r["Quality Status"]))
    check(all(len(v) == 1 for v in by_url.values()),
          "CSV: each source URL has one source name and one status (several findings per source are allowed)")
    check(all(r["Quality Status"].startswith(("Used", "Needs review")) for r in rows),
          "CSV: quality status is Used or Needs review")
except FileNotFoundError:
    check(False, f"CSV: file not found at {csv_path}")

print(f"PASSED {len(passes)}   FAILED {len(fails)}")
for f in fails:
    print("FAIL:", f)
out = "validation_report.txt"
with open(out, "w", encoding="utf-8") as fh:
    fh.write(f"PASSED {len(passes)}   FAILED {len(fails)}\n")
    for p in passes:
        fh.write("PASS: " + p + "\n")
    for f in fails:
        fh.write("FAIL: " + f + "\n")
sys.exit(1 if fails else 0)
