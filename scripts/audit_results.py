"""
Benchmark Results Audit — 5-Phase Research Investigation
=========================================================
Audits S2 and S3 benchmark results for score inflation, truncation,
coverage check failures, and statistical anomalies.

Usage:
    venv/bin/python3 scripts/audit_results.py
"""

import json
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "benchmark.db"
DATA_DIR = PROJECT_ROOT / "data"

_SEVERITY_PENALTY_CORRECT = {"critical": 0.25, "major": 0.05, "minor": 0.01}
TARGET_SCENARIOS = {"s2_gemini_flash_zeroshot", "s3_gemini_pro_dspy_predict"}

# Full expected eval set from dataset.py
_EXPECTED_DOC_IDS = (
    [f"Form_{i:03d}" for i in range(1, 11)] +
    [f"Summary_{i:03d}" for i in range(1, 11)] +
    [f"Prescript_{i:03d}" for i in range(35, 45)] +
    [f"Referral_{i:03d}" for i in range(68, 78)]
)

SEP = "─" * 72


def _divider(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _recalc_score(errors: list[dict]) -> float:
    total_penalty = sum(
        _SEVERITY_PENALTY_CORRECT.get(e.get("severity", "").lower(), 0.0)
        for e in errors
    )
    return max(0.0, 1.0 - total_penalty)


def _load_results(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("""
        SELECT r.id, r.doc_id, s.name AS scenario, r.quality_score,
               r.errors, r.translation, r.input_tokens, r.output_tokens,
               r.elapsed_sec
        FROM results r
        JOIN scenarios s ON r.scenario_id = s.id
        WHERE s.name IN ('s2_gemini_flash_zeroshot', 's3_gemini_pro_dspy_predict')
        ORDER BY s.name, r.doc_id
    """).fetchall()
    results = []
    for row in rows:
        r = dict(row)
        r["errors_parsed"] = json.loads(r["errors"]) if r["errors"] else []
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Phase 1 — Severity Case-Sensitivity Bug
# ---------------------------------------------------------------------------

def phase1_severity_bug(results: list[dict]) -> None:
    _divider("PHASE 1 — Severity Case-Sensitivity Bug (llm_judge.py line 212)")

    affected = []
    clean = []

    for r in results:
        errors = r["errors_parsed"]
        stored = r["quality_score"]
        corrected = _recalc_score(errors)
        delta = stored - corrected

        # Check for capitalized severity values
        severities = [e.get("severity", "") for e in errors]
        has_wrong_case = any(s != s.lower() for s in severities)

        if has_wrong_case or abs(delta) > 1e-9:
            affected.append({**r, "corrected": corrected, "delta": delta, "severities": severities})
        else:
            clean.append(r)

    # Aggregate stats
    s2_results = [r for r in results if r["scenario"] == "s2_gemini_flash_zeroshot"]
    s3_results = [r for r in results if r["scenario"] == "s3_gemini_pro_dspy_predict"]

    def avg(lst, key): return sum(x[key] for x in lst) / len(lst) if lst else 0.0

    s2_stored_avg = avg(s2_results, "quality_score")
    s3_stored_avg = avg(s3_results, "quality_score")
    s2_corrected_avg = sum(_recalc_score(r["errors_parsed"]) for r in s2_results) / len(s2_results) if s2_results else 0.0
    s3_corrected_avg = sum(_recalc_score(r["errors_parsed"]) for r in s3_results) / len(s3_results) if s3_results else 0.0

    print(f"\nTotal results audited: {len(results)} (S2={len(s2_results)}, S3={len(s3_results)})")
    print(f"Results with case-sensitivity bug or score delta: {len(affected)}")
    print(f"Results unaffected (severity already lowercase): {len(clean)}")

    print(f"\n{'Scenario':<35} {'Stored Avg':>12} {'Corrected Avg':>14} {'Delta':>8}")
    print(f"{'─'*35} {'─'*12} {'─'*14} {'─'*8}")
    print(f"{'s2_gemini_flash_zeroshot':<35} {s2_stored_avg:>12.4f} {s2_corrected_avg:>14.4f} {s2_stored_avg - s2_corrected_avg:>+8.4f}")
    print(f"{'s3_gemini_pro_dspy_predict':<35} {s3_stored_avg:>12.4f} {s3_corrected_avg:>14.4f} {s3_stored_avg - s3_corrected_avg:>+8.4f}")

    if affected:
        print(f"\n{'Doc ID':<18} {'Scenario':<28} {'Stored':>8} {'Corrected':>10} {'Delta':>8} {'Severities in errors'}")
        print(f"{'─'*18} {'─'*28} {'─'*8} {'─'*10} {'─'*8} {'─'*30}")
        for r in sorted(affected, key=lambda x: abs(x["delta"]), reverse=True):
            sevs = ", ".join(r["severities"]) if r["severities"] else "(none)"
            print(f"{r['doc_id']:<18} {r['scenario']:<28} {r['quality_score']:>8.4f} {r['corrected']:>10.4f} {r['delta']:>+8.4f}  {sevs}")
    else:
        print("\nNo affected results found — severities are already lowercase OR all errors produce correct scores.")

    # Show all distinct severity values seen in the DB
    all_severities = []
    for r in results:
        all_severities.extend(e.get("severity", "") for e in r["errors_parsed"])
    distinct = sorted(set(all_severities))
    print(f"\nAll distinct severity values found in DB: {distinct}")


# ---------------------------------------------------------------------------
# Phase 2 — Perfect Score Manual Review
# ---------------------------------------------------------------------------

def phase2_perfect_scores(results: list[dict]) -> None:
    _divider("PHASE 2 — Perfect Score Manual Review (score >= 0.99)")

    perfect = [r for r in results if r["quality_score"] >= 0.99]
    print(f"\n{len(perfect)} results with score >= 0.99:\n")

    print(f"{'Doc ID':<18} {'Scenario':<28} {'Score':>7} {'Errors':>7} {'Trans Len':>10} {'Pairs':>7} {'Src Len':>8} {'Ratio':>7}")
    print(f"{'─'*18} {'─'*28} {'─'*7} {'─'*7} {'─'*10} {'─'*7} {'─'*8} {'─'*7}")

    issues_found = []
    for r in sorted(perfect, key=lambda x: (x["scenario"], x["doc_id"])):
        errors = r["errors_parsed"]
        trans_raw = r["translation"]
        trans_len = len(trans_raw)

        # Parse translation pairs
        try:
            pairs = json.loads(trans_raw)
            num_pairs = len(pairs) if isinstance(pairs, list) else "?"
        except Exception:
            pairs = []
            num_pairs = "PARSE_ERR"

        # Read source file for length comparison
        doc_id = r["doc_id"]
        prefix = doc_id.rsplit("_", 1)[0]  # e.g. "Form"
        src_path = _find_source_file(doc_id)
        src_len = len(src_path.read_text(encoding="utf-8")) if src_path else -1
        ratio = f"{trans_len/src_len:.2f}x" if src_len > 0 else "N/A"

        print(f"{doc_id:<18} {r['scenario']:<28} {r['quality_score']:>7.4f} {len(errors):>7} {trans_len:>10,} {num_pairs!s:>7} {src_len:>8,} {ratio:>7}")

        # Flag anomalies
        flags = []
        if src_len > 0 and trans_len / src_len > 4.0:
            flags.append(f"TRANSLATION_VERY_LONG ({trans_len/src_len:.1f}x source)")
        if src_len > 0 and trans_len / src_len < 1.2:
            flags.append(f"TRANSLATION_VERY_SHORT ({trans_len/src_len:.2f}x source)")
        if isinstance(pairs, list) and isinstance(num_pairs, int) and src_len > 0:
            # rough sentences in source (split by newline as proxy)
            src_text = src_path.read_text(encoding="utf-8")
            src_lines = [l.strip() for l in src_text.split("\n") if l.strip()]
            coverage = num_pairs / len(src_lines) if src_lines else 1.0
            if coverage < 0.8:
                flags.append(f"LOW_COVERAGE ({num_pairs} pairs vs {len(src_lines)} src lines = {coverage:.0%})")
        if errors:
            sev_vals = [e.get("severity", "") for e in errors]
            flags.append(f"HAS_ERRORS_BUT_PERFECT: {sev_vals}")

        if flags:
            issues_found.append((doc_id, r["scenario"], flags))

    if issues_found:
        print("\n  ⚠  ANOMALIES IN PERFECT-SCORE RESULTS:")
        for doc_id, scenario, flags in issues_found:
            print(f"  • {doc_id} ({scenario}):")
            for f in flags:
                print(f"      - {f}")
    else:
        print("\n  No structural anomalies found in perfect-score results.")

    # Spot-check: show first pair and last pair of each perfect-score translation
    print("\n--- Spot-check: first and last sentence pairs of perfect-score results ---")
    for r in sorted(perfect, key=lambda x: (x["scenario"], x["doc_id"])):
        try:
            pairs = json.loads(r["translation"])
            if isinstance(pairs, list) and pairs:
                first = pairs[0]
                last = pairs[-1]
                print(f"\n  [{r['doc_id']} / {r['scenario']}] — {len(pairs)} pairs")
                print(f"    FIRST src:  {str(first.get('source_hebrew',''))[:80]}")
                print(f"    FIRST tgt:  {str(first.get('translated_russian',''))[:80]}")
                print(f"    LAST  src:  {str(last.get('source_hebrew',''))[:80]}")
                print(f"    LAST  tgt:  {str(last.get('translated_russian',''))[:80]}")
        except Exception as e:
            print(f"  [{r['doc_id']}] Parse error: {e}")


def _find_source_file(doc_id: str) -> Path | None:
    """Locate the Hebrew source .txt file for a given doc_id."""
    prefix = doc_id.split("_")[0].lower()
    subdir_map = {
        "form": DATA_DIR / "informed_consent_forms" / "text" / "he",
        "summary": DATA_DIR / "summaries",
        "prescript": DATA_DIR / "prescripts",
        "referral": DATA_DIR / "referrals",
    }
    subdir = subdir_map.get(prefix)
    if subdir is None:
        return None
    path = subdir / f"{doc_id}_HE.txt"
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Phase 3 — Score Distribution & Error Category Analysis
# ---------------------------------------------------------------------------

def phase3_distribution(results: list[dict]) -> None:
    _divider("PHASE 3 — Score Distribution & Error Category Analysis")

    # Score distribution by scenario + doc type
    print("\n--- Score distribution by scenario and document type ---\n")
    print(f"{'Scenario':<28} {'DocType':<12} {'N':>4} {'Avg':>7} {'Min':>7} {'Max':>7} {'Perfect':>8} {'>=0.99':>7}")
    print(f"{'─'*28} {'─'*12} {'─'*4} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*7}")

    from collections import defaultdict
    buckets: dict[tuple, list] = defaultdict(list)
    for r in results:
        doc_type = r["doc_id"].split("_")[0]
        buckets[(r["scenario"], doc_type)].append(r["quality_score"])

    for (scenario, doc_type), scores in sorted(buckets.items()):
        n = len(scores)
        avg = sum(scores) / n
        mn = min(scores)
        mx = max(scores)
        perfect = sum(1 for s in scores if s == 1.0)
        near = sum(1 for s in scores if s >= 0.99)
        print(f"{scenario:<28} {doc_type:<12} {n:>4} {avg:>7.4f} {mn:>7.4f} {mx:>7.4f} {perfect:>8} {near:>7}")

    # Error category/severity breakdown
    print("\n--- Error frequency by category and severity ---\n")
    cat_sev: dict[tuple, int] = defaultdict(int)
    for r in results:
        for e in r["errors_parsed"]:
            cat = e.get("category", "unknown")
            sev = e.get("severity", "unknown")
            cat_sev[(r["scenario"], cat, sev)] += 1

    print(f"{'Scenario':<28} {'Category':<28} {'Severity':<10} {'Count':>6}")
    print(f"{'─'*28} {'─'*28} {'─'*10} {'─'*6}")
    for (scenario, cat, sev), count in sorted(cat_sev.items(), key=lambda x: -x[1]):
        print(f"{scenario:<28} {cat:<28} {sev:<10} {count:>6}")

    # Score histogram (buckets of 0.05)
    print("\n--- Score histogram (all S2+S3 results) ---\n")
    buckets_hist: dict[str, int] = defaultdict(int)
    for r in results:
        bucket = f"{(int(r['quality_score'] * 20)) / 20:.2f}–{(int(r['quality_score'] * 20) + 1) / 20:.2f}"
        buckets_hist[bucket] += 1
    for b, count in sorted(buckets_hist.items()):
        bar = "█" * count
        print(f"  {b:>12}  {bar:<30} {count}")

    # Zero-error results (potential concern)
    zero_error = [r for r in results if not r["errors_parsed"]]
    print(f"\nResults with zero errors recorded: {len(zero_error)} / {len(results)}")
    if zero_error:
        print(f"  These are:")
        for r in sorted(zero_error, key=lambda x: (x["scenario"], x["doc_id"])):
            print(f"    {r['doc_id']:<18} {r['scenario']:<28}  score={r['quality_score']:.4f}")


# ---------------------------------------------------------------------------
# Phase 4 — Missing S2 Documents
# ---------------------------------------------------------------------------

def phase4_missing_s2(results: list[dict]) -> None:
    _divider("PHASE 4 — Missing S2 Documents")

    s2_doc_ids = {r["doc_id"] for r in results if r["scenario"] == "s2_gemini_flash_zeroshot"}
    missing = [d for d in _EXPECTED_DOC_IDS if d not in s2_doc_ids]

    print(f"\nExpected S2 docs: {len(_EXPECTED_DOC_IDS)}")
    print(f"Present S2 docs:  {len(s2_doc_ids)}")
    print(f"Missing S2 docs:  {len(missing)}")

    if missing:
        print(f"\nMissing doc IDs: {missing}")
        print("\n--- Source file sizes for missing docs ---")
        for doc_id in missing:
            src = _find_source_file(doc_id)
            if src:
                size = src.stat().st_size
                text = src.read_text(encoding="utf-8")
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                print(f"  {doc_id:<18}  {size:>8,} bytes  ~{len(lines)} non-empty lines")
            else:
                print(f"  {doc_id:<18}  source file NOT FOUND")

        print("\n--- Comparison: sizes of S2 completed docs ---")
        completed_sizes = []
        for r in results:
            if r["scenario"] == "s2_gemini_flash_zeroshot":
                src = _find_source_file(r["doc_id"])
                if src:
                    completed_sizes.append((r["doc_id"], src.stat().st_size, r["quality_score"]))
        completed_sizes.sort(key=lambda x: -x[1])
        print(f"  Largest 5 completed docs (by source size):")
        for doc_id, sz, score in completed_sizes[:5]:
            print(f"    {doc_id:<18}  {sz:>8,} bytes  score={score:.4f}")
        print(f"  Smallest 5 completed docs (by source size):")
        for doc_id, sz, score in completed_sizes[-5:]:
            print(f"    {doc_id:<18}  {sz:>8,} bytes  score={score:.4f}")

    # Bias assessment: would missing docs likely be hard cases?
    print("\n--- Bias assessment: are missing docs outliers? ---")
    s2_scores = [r["quality_score"] for r in results if r["scenario"] == "s2_gemini_flash_zeroshot"]
    print(f"  Current S2 avg (37 docs): {sum(s2_scores)/len(s2_scores):.4f}")
    print(f"  Note: Timeout docs likely had long/complex source text → likely harder to translate")
    print(f"  → S2 average is likely inflated upward by excluding these 3 problematic docs.")


# ---------------------------------------------------------------------------
# Phase 5 — Translation Length & Coverage Plausibility
# ---------------------------------------------------------------------------

def phase5_translation_lengths(results: list[dict]) -> None:
    _divider("PHASE 5 — Translation Length & Coverage Plausibility")

    print(f"\n{'Doc ID':<18} {'Scenario':<28} {'Score':>7} {'Src Len':>8} {'Trs Len':>8} {'Ratio':>7} {'#Pairs':>7} {'Flags'}")
    print(f"{'─'*18} {'─'*28} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7} {'─'*30}")

    anomalies = []
    for r in sorted(results, key=lambda x: (x["doc_id"], x["scenario"])):
        src = _find_source_file(r["doc_id"])
        src_len = len(src.read_text(encoding="utf-8")) if src else -1
        trs_len = len(r["translation"])
        ratio = trs_len / src_len if src_len > 0 else 0.0

        try:
            pairs = json.loads(r["translation"])
            num_pairs = len(pairs) if isinstance(pairs, list) else "?"
        except Exception:
            pairs = []
            num_pairs = "ERR"

        flags = []
        if ratio > 5.0:
            flags.append(f"VERY_LONG({ratio:.1f}x)")
        elif ratio > 3.5:
            flags.append(f"LONG({ratio:.1f}x)")
        if ratio < 1.5 and src_len > 0:
            flags.append(f"SHORT({ratio:.2f}x)")

        # Check JSON completeness
        trans_stripped = r["translation"].strip()
        if not trans_stripped.endswith("]") and not trans_stripped.endswith("}"):
            flags.append("JSON_TRUNCATED?")

        flag_str = " | ".join(flags) if flags else ""
        if flags:
            anomalies.append((r["doc_id"], r["scenario"], flags))

        print(f"{r['doc_id']:<18} {r['scenario']:<28} {r['quality_score']:>7.4f} {src_len:>8,} {trs_len:>8,} {ratio:>7.2f} {num_pairs!s:>7} {flag_str}")

    if anomalies:
        print(f"\n  ⚠  FLAGGED RESULTS ({len(anomalies)} total):")
        for doc_id, scenario, flags in anomalies:
            print(f"  • {doc_id} ({scenario}): {' | '.join(flags)}")
    else:
        print("\n  No translation length anomalies detected.")

    # Summary statistics
    ratios = []
    for r in results:
        src = _find_source_file(r["doc_id"])
        if src:
            src_len = len(src.read_text(encoding="utf-8"))
            if src_len > 0:
                ratios.append(len(r["translation"]) / src_len)
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\nTranslation length ratio stats (translation_chars / source_chars):")
        print(f"  Mean:  {avg_ratio:.2f}x")
        print(f"  Min:   {min(ratios):.2f}x")
        print(f"  Max:   {max(ratios):.2f}x")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\n" + "═" * 72)
    print("  MEDIFLOW BENCHMARK AUDIT — S2 & S3 RESULTS")
    print("═" * 72)

    results = _load_results(conn)
    print(f"\nLoaded {len(results)} results from benchmark.db")

    phase1_severity_bug(results)
    phase2_perfect_scores(results)
    phase3_distribution(results)
    phase4_missing_s2(results)
    phase5_translation_lengths(results)

    print(f"\n{'═'*72}")
    print("  AUDIT COMPLETE")
    print("═" * 72 + "\n")

    conn.close()


if __name__ == "__main__":
    main()
