from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir


DEFAULT_YEAR_PATTERN = r"\b(18\d{2}|19\d{2}|200\d|201\d|202[0-6])\b"


def normalize_case_id(raw_id: object) -> str:
    case_id = str(raw_id).strip()
    if case_id.endswith(".txt"):
        case_id = case_id[:-4]
    return case_id


def load_ids(path: str | Path) -> List[str]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        ids = [normalize_case_id(line) for line in f if line.strip()]
    # Keep order but remove duplicates.
    deduped: List[str] = []
    seen: set[str] = set()
    for case_id in ids:
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        deduped.append(case_id)
    return deduped


def resolve_case_path(case_id: str, directory: Path) -> Path | None:
    candidates = [directory / f"{case_id}.txt", directory / f"{case_id.zfill(6)}.txt"]
    return next((path for path in candidates if path.exists()), None)


def extract_max_year(text: str, pattern: re.Pattern[str]) -> int:
    years = [int(year) for year in pattern.findall(text)]
    return max(years, default=0)


def collect_case_paths(
    directory: Path,
    *,
    selected_ids: Iterable[str] | None = None,
) -> tuple[Dict[str, Path], List[str]]:
    found: Dict[str, Path] = {}
    missing: List[str] = []
    if selected_ids is None:
        for path in sorted(directory.glob("*.txt")):
            if path.is_file():
                found[path.stem] = path
        return found, missing

    for case_id in selected_ids:
        path = resolve_case_path(case_id, directory)
        if path is None:
            missing.append(case_id)
            continue
        found[normalize_case_id(case_id)] = path
    return found, missing


def build_year_index(case_paths: Dict[str, Path], year_pattern: re.Pattern[str]) -> Dict[str, int]:
    year_index: Dict[str, int] = {}
    for case_id, path in case_paths.items():
        text = path.read_text(encoding="utf-8", errors="ignore")
        year_index[case_id] = extract_max_year(text, year_pattern)
    return year_index


def build_scope(
    query_years: Dict[str, int],
    ordered_candidate_ids: List[str],
    candidate_years: Dict[str, int],
    *,
    unknown_query_year_policy: str,
    exclude_self: bool,
) -> Dict[str, List[str]]:
    scope: Dict[str, List[str]] = {}
    candidate_year_values = [candidate_years[candidate_id] for candidate_id in ordered_candidate_ids]

    for qid, qyear in query_years.items():
        if qyear == 0:
            allowed = list(ordered_candidate_ids) if unknown_query_year_policy == "all" else []
        else:
            allowed = [
                candidate_id
                for candidate_id, candidate_year in zip(ordered_candidate_ids, candidate_year_values)
                if candidate_year <= qyear
            ]
        if exclude_self:
            allowed = [candidate_id for candidate_id in allowed if candidate_id != qid]
        scope[qid] = allowed
    return scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a query->candidate scope index by filtering candidates with "
            "year(candidate) <= year(query)."
        )
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Directory containing candidate `.txt` cases.",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        required=True,
        help="Directory containing query `.txt` cases.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output JSON path for query->candidate scope mapping.",
    )
    parser.add_argument(
        "--query-ids-path",
        type=Path,
        default=None,
        help="Optional file with one query id per line. If omitted, use all files in query-dir.",
    )
    parser.add_argument(
        "--candidate-ids-path",
        type=Path,
        default=None,
        help="Optional file with one candidate id per line. If omitted, use all files in candidate-dir.",
    )
    parser.add_argument(
        "--year-pattern",
        type=str,
        default=DEFAULT_YEAR_PATTERN,
        help=f"Regex used to extract years. Default: {DEFAULT_YEAR_PATTERN}",
    )
    parser.add_argument(
        "--unknown-query-year-policy",
        choices=["all", "empty"],
        default="all",
        help="When query year is not found: keep all candidates or keep none.",
    )
    parser.add_argument(
        "--exclude-self",
        action="store_true",
        help="Exclude candidate with the same id as query id.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indent size for output JSON. Use 0 for compact JSON.",
    )
    return parser.parse_args()


def _execute(args: argparse.Namespace) -> None:
    candidate_dir: Path = args.candidate_dir
    query_dir: Path = args.query_dir

    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_dir}")
    if not query_dir.is_dir():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")

    year_pattern = re.compile(args.year_pattern)

    selected_candidate_ids = load_ids(args.candidate_ids_path) if args.candidate_ids_path else None
    selected_query_ids = load_ids(args.query_ids_path) if args.query_ids_path else None

    candidate_paths, missing_candidate_ids = collect_case_paths(
        candidate_dir,
        selected_ids=selected_candidate_ids,
    )
    query_paths, missing_query_ids = collect_case_paths(
        query_dir,
        selected_ids=selected_query_ids,
    )

    if not candidate_paths:
        raise RuntimeError("No candidate files found.")
    if not query_paths:
        raise RuntimeError("No query files found.")

    print(f"Candidates found: {len(candidate_paths)}")
    print(f"Queries found: {len(query_paths)}")
    if missing_candidate_ids:
        print(f"Missing candidate IDs from list: {len(missing_candidate_ids)} (example: {missing_candidate_ids[:5]})")
    if missing_query_ids:
        print(f"Missing query IDs from list: {len(missing_query_ids)} (example: {missing_query_ids[:5]})")

    print("Extracting candidate years...")
    candidate_years = build_year_index(candidate_paths, year_pattern)
    print("Extracting query years...")
    query_years = build_year_index(query_paths, year_pattern)

    ordered_candidate_ids = sorted(candidate_paths.keys())
    # Keep query order from provided IDs when available; otherwise sort by id.
    if selected_query_ids is not None:
        ordered_query_ids = [qid for qid in selected_query_ids if qid in query_years]
    else:
        ordered_query_ids = sorted(query_years.keys())
    ordered_query_years = {qid: query_years[qid] for qid in ordered_query_ids}

    print("Building query->candidate scope index...")
    scope = build_scope(
        ordered_query_years,
        ordered_candidate_ids,
        candidate_years,
        unknown_query_year_policy=args.unknown_query_year_policy,
        exclude_self=args.exclude_self,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        if args.indent <= 0:
            json.dump(scope, f, ensure_ascii=False)
        else:
            json.dump(scope, f, ensure_ascii=False, indent=args.indent)

    candidate_count = len(ordered_candidate_ids)
    avg_candidates = sum(len(ids) for ids in scope.values()) / max(len(scope), 1)
    unknown_query_year_count = sum(1 for year in ordered_query_years.values() if year == 0)
    unknown_candidate_year_count = sum(1 for year in candidate_years.values() if year == 0)

    print(f"Saved scope index to: {args.output_path}")
    print(f"Candidate count: {candidate_count}")
    print(f"Average candidates per query: {avg_candidates:.2f}")
    print(f"Queries without extracted year: {unknown_query_year_count}/{len(ordered_query_years)}")
    print(f"Candidates without extracted year: {unknown_candidate_year_count}/{candidate_count}")


def _build_default_args_from_repo_root() -> argparse.Namespace:
    """
    Default configuration for running directly from this repository:
      python "Legal Case Retrieval/pre-process/build_query_candidate_scope.py"
    """
    task1_root = Path(get_task1_dir())

    candidate_dir = task1_root / "processed"
    query_dir = task1_root / "processed"
    output_path = task1_root / "lht_process" / "modernBert" / "query_candidate_scope.json"
    train_qids_path = task1_root / "train_qid.tsv"
    valid_qids_path = task1_root / "valid_qid.tsv"

    if not train_qids_path.exists():
        raise FileNotFoundError(f"Default train qid file not found: {train_qids_path}")
    if not valid_qids_path.exists():
        raise FileNotFoundError(f"Default valid qid file not found: {valid_qids_path}")

    # Combine train+valid query IDs while preserving order and removing duplicates.
    merged_qids: List[str] = []
    seen: set[str] = set()
    for src_path in (train_qids_path, valid_qids_path):
        for qid in load_ids(src_path):
            if qid in seen:
                continue
            seen.add(qid)
            merged_qids.append(qid)

    query_ids_path = Path("/tmp/train_valid_qids_all.tsv")
    query_ids_path.write_text("\n".join(merged_qids) + "\n", encoding="utf-8")

    return argparse.Namespace(
        candidate_dir=candidate_dir,
        query_dir=query_dir,
        output_path=output_path,
        query_ids_path=query_ids_path,
        candidate_ids_path=None,
        year_pattern=DEFAULT_YEAR_PATTERN,
        unknown_query_year_policy="all",
        exclude_self=False,
        indent=2,
    )


def main() -> None:
    # No extra args => use project default config.
    # Any extra args => keep CLI override behavior.
    if len(sys.argv) == 1:
        args = _build_default_args_from_repo_root()
        print("Using built-in default configuration (no CLI args provided).")
    else:
        args = parse_args()
    _execute(args)


if __name__ == "__main__":
    main()
