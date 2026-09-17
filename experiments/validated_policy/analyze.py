"""Validate complete paired runs and report terminal-score gains (standard library)."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    """Read a CSV without discarding failed simulations."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def interval(values):
    """Normal approximate 95% interval; random cohorts pass deck-level means."""
    mean = statistics.mean(values)
    if len(values) < 2:
        return None
    radius = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - radius, mean + radius]


def analyze(manifest_path, csv_path):
    """Reject missing/duplicate/error rows and verify each manifest pairing."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = read_rows(csv_path)
    cases = {c["name"]: c for c in manifest["cases"]}
    lookup = {(r["case"], int(r["run"]), r["variant"]): r for r in rows}
    expected = {(c, run, variant) for c in cases for run in range(manifest["runs"])
        for variant in manifest["variants"]}
    if len(lookup) != len(rows) or set(lookup) != expected or any(r["error"] for r in rows):
        raise ValueError("Missing, duplicate, unexpected, or failed simulations")
    for row in rows:
        case = cases[row["case"]]
        identity = (row["cohort"], row["uma"], row["deck"], row["base_seed"])
        expected_identity = (case["cohort"], str(case["uma"]), "/".join(map(str, case["deck"])), str(case["seed"]))
        base = lookup[row["case"], int(row["run"]), "base"]
        if identity != expected_identity or row["rule_seed"] != base["rule_seed"]:
            raise ValueError("Manifest identity or paired rule seed mismatch")
    results = {}
    refs = ["base", "ptblend200", "ptblend200-capd0-rgn1", "ptblend200-capd0-rgn1-supermode3"]
    for variant in manifest["variants"]:
        if variant == "base":
            continue
        results[variant] = {}
        for ref in refs:
            if ref == variant or ref not in manifest["variants"]:
                continue
            groups = {}
            for cohort in ["fixed", "preset", "random", "random_diverse", "random_narrow"]:
                selected = [r for r in rows if r["variant"] == variant and
                    (r["cohort"] == cohort or cohort == "random" and r["cohort"].startswith("random"))]
                if not selected:
                    continue
                diffs, by_deck = [], defaultdict(list)
                for row in selected:
                    baseline = lookup[row["case"], int(row["run"]), ref]
                    diff = int(row["score"]) - int(baseline["score"])
                    diffs.append(diff)
                    key = row["deck"] if cohort.startswith("random") else row["case"]
                    by_deck[key].append(diff)
                means = [statistics.mean(v) for v in by_deck.values()]
                groups[cohort] = dict(pairs=len(diffs), decks=len(means), delta=statistics.mean(diffs),
                    ci=interval(means if cohort.startswith("random") else diffs),
                    negative_decks=sum(v < 0 for v in means), worst_deck=min(means))
            results[variant][ref] = groups
    output = dict(manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        rows=len(rows), comparisons=results)
    Path(csv_path).with_suffix(".summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Complete paired simulations:", len(rows))
    for variant, refs in results.items():
        for ref, groups in refs.items():
            if variant not in ["ptblend200-capd0-rgn1", "ptblend200-capd0-rgn1-supermode3",
                "ptblend200-capd0-rgn1-supermode3-hintlv600"]:
                continue
            print(variant, "vs", ref)
            for cohort in ["fixed", "preset", "random"]:
                if cohort in groups:
                    g = groups[cohort]
                    print(" ", cohort, round(g["delta"], 2), "CI", g["ci"])


def verify(expected_path, actual_path):
    """Compare frozen result fields exactly, ignoring only run time and row order."""
    def normalized(path):
        rows = read_rows(path)
        keys = [(r["case"], r["run"], r["variant"]) for r in rows]
        if len(set(keys)) != len(rows):
            raise ValueError("Duplicate replay rows")
        return sorted([{k: v for k, v in r.items() if k != "elapsed_ms"} for r in rows],
            key=lambda r: (r["case"], int(r["run"]), r["variant"]))
    expected, actual = normalized(expected_path), normalized(actual_path)
    if not expected or expected != actual:
        raise ValueError("Replay differs from frozen expected output")
    print("Frozen replay matches:", len(actual), "rows (timing excluded)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["analyze", "verify"])
    parser.add_argument("input", help="Manifest JSON for analyze; expected CSV for verify")
    parser.add_argument("csv", help="Actual simulation CSV")
    args = parser.parse_args()
    if args.action == "analyze":
        analyze(args.input, args.csv)
    else:
        verify(args.input, args.csv)
