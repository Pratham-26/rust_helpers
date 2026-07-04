"""Profile polars-stringsim on the FEBRL4 record-linkage dataset.

FEBRL4 (Freely Extensible Biomedical Record Linkage, dataset 4) is the
canonical synthetic benchmark: two files of 5000 records each, 5000 true
links (one duplicate per original), with name/address fields and injected
noise (typos, missing values, swapped tokens).

This harness measures:
  1. Per-metric expression timing on a fixed pair sample.
  2. hybrid_score (single Rust call) vs Python combine() over N exprs.
  3. fuzzy_join scaling at 1k / 2k / 5k records, with and without blocking.
  4. deduplicate timing on one frame.
  5. Accuracy (precision/recall/F1) of fuzzy_join vs ground truth, full
     cross on a 1000x1000 subsample so no true match is dropped by blocking.

Usage:
    python profiling/profile_fuzzy.py
"""
from __future__ import annotations

import time
from typing import Callable

import polars as pl
import polars_stringsim as pf
from recordlinkage.datasets import load_febrl4

REPEAT = 3  # repetitions per timed call (take min)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_febrl() -> tuple[pl.DataFrame, pl.DataFrame, set[tuple[str, str]]]:
    """Return (frame_a, frame_b, true_links) as Polars frames.

    Each frame has columns: rec_id, name (given_name + " " + surname).
    true_links is a set of (rec_id_a, rec_id_b).
    """
    a_pd, b_pd, links = load_febrl4(return_links=True)

    def to_polars(df, id_label):
        out = pl.from_pandas(df[["given_name", "surname"]].fillna(""))
        out = out.with_columns(
            name=pl.col("given_name") + pl.lit(" ") + pl.col("surname"),
            rec_id=pl.Series(df.index.astype(str)),
        ).select("rec_id", "name")
        return out.rename({"rec_id": id_label})

    a = to_polars(a_pd, "rec_id_a")
    b = to_polars(b_pd, "rec_id_b")
    true = set(links)
    return a, b, true


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------

def time_it(fn: Callable, label: str, repeat: int = REPEAT) -> float:
    """Run `fn` `repeat` times, return the min wall-clock seconds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def fmt(secs: float) -> str:
    if secs >= 1.0:
        return f"{secs:7.3f} s "
    return f"{secs * 1000:7.2f} ms"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n {title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

# All registry metric names that take exactly (left, right) with no extra args.
SINGLE_ARG_METRICS = [
    "jaro", "jaro_winkler",
    "levenshtein_norm", "damerau_levenshtein_norm", "hamming_norm",
    "token_jaccard", "token_sorensen_dice",
    "trigram_jaccard", "trigram_sorensen_dice",
    "lcs_sim",
    "soundex_jw_sim", "metaphone_jw_sim", "double_metaphone_jw_sim", "nysiis_jw_sim",
]


def bench_single_metrics(pairs: pl.DataFrame) -> None:
    """Time each single-arg metric expression over a fixed pair frame."""
    banner(f"1. Per-metric expression timing ({pairs.height:,} pairs)")
    print(f"{'metric':<22} {'time':>10}    {'pairs/sec':>14}")
    print("-" * 60)
    for name in SINGLE_ARG_METRICS:
        builder = getattr(pf, name, None)
        if builder is None:
            # try registry-resolved name
            continue

        def run(_b=builder):
            pairs.with_columns(score=_b("a", "b"))

        secs = time_it(run, name)
        rate = pairs.height / secs
        print(f"{name:<22} {fmt(secs):>10}    {rate:>14,.0f}")


def bench_hybrid_vs_combine(pairs: pl.DataFrame) -> None:
    """Compare hybrid_score (1 Rust call) to Python combine() over N exprs."""
    banner("2. hybrid_score (Rust) vs Python combine() over separate exprs")
    algos = ["jaro_winkler", "trigram_jaccard", "token_jaccard"]
    weights = [0.5, 0.3, 0.2]

    def run_hybrid():
        pairs.with_columns(
            score=pf.hybrid_score("a", "b", algorithms=algos, weights=weights)
        )

    # Pre-build metric exprs for combine.
    metrics = [pf.jaro_winkler("a", "b"),
               pf.trigram_jaccard("a", "b"),
               pf.token_jaccard("a", "b")]

    def run_combine():
        pairs.with_columns(
            score=pf.combine(metrics, weights=weights, method="weighted_avg")
        )

    t_h = time_it(run_hybrid, "hybrid")
    t_c = time_it(run_combine, "combine")
    print(f"{'hybrid_score (3 algos)':<28} {fmt(t_h):>10}")
    print(f"{'combine (3 separate exprs)':<28} {fmt(t_c):>10}")
    print(f"{'speedup of hybrid':<28} {t_c / t_h:>9.2f}x")


def bench_fuzzy_join_scaling(a: pl.DataFrame, b: pl.DataFrame) -> None:
    banner("3. fuzzy_join scaling (full cross vs first_chars blocking)")
    sizes = [1000, 2000, 5000]
    print(f"{'config':<34} {'pairs':>10} {'time':>10} {'pairs/sec':>14}")
    print("-" * 72)

    for n in sizes:
        la = a.head(n)
        rb = b.head(n)

        # Full cross — only feasible for small n.
        if n <= 2000:
            def run_cross(_l=la, _r=rb):
                pf.fuzzy_join(
                    _l, _r, left_on="name", right_on="name",
                    algorithms=["jaro_winkler", "trigram_jaccard"],
                    threshold=0.7,
                )
            secs = time_it(run_cross, f"cross {n}")
            print(f"{'full cross n=' + str(n):<34} {n*n:>10,} {fmt(secs):>10} "
                  f"{n*n/secs:>14,.0f}")

        def run_blocked(_l=la, _r=rb):
            pf.fuzzy_join(
                _l, _r, left_on="name", right_on="name",
                algorithms=["jaro_winkler", "trigram_jaccard"],
                threshold=0.7, block="first_chars", block_n=2,
            )
        secs = time_it(run_blocked, f"blocked {n}")
        # estimate blocked pairs by collecting once
        sample = pf.fuzzy_join(
            la, rb, left_on="name", right_on="name",
            algorithms=["jaro_winkler", "trigram_jaccard"],
            threshold=0.0, block="first_chars", block_n=2,
        )
        npairs = sample.height
        print(f"{'blocked(2-char) n=' + str(n):<34} {npairs:>10,} {fmt(secs):>10} "
              f"{npairs/secs if secs > 0 else 0:>14,.0f}")


def bench_deduplicate(a: pl.DataFrame) -> None:
    banner("4. deduplicate timing (single frame, blocked)")
    sizes = [1000, 2000, 5000]
    print(f"{'n':>6} {'time':>10}")
    print("-" * 22)
    for n in sizes:
        sub = a.head(n)
        secs = time_it(
            lambda _s=sub: pf.deduplicate(
                _s, on="name", composite_threshold=0.85,
                algorithms=["jaro_winkler", "trigram_jaccard"],
                block="first_chars", block_n=2,
            ),
            f"dedup {n}",
        )
        print(f"{n:>6} {fmt(secs):>10}")


def bench_accuracy(a: pl.DataFrame, b: pl.DataFrame, true: set) -> None:
    """precision/recall/F1 of fuzzy_join vs ground truth on a full-cross subsample."""
    banner("5. Accuracy vs ground truth (full cross, n=1000 each side)")
    n = 1000
    la = a.head(n)
    rb = b.head(n)

    # Build pairs via fuzzy_join with a low threshold so we can sweep.
    threshold = 0.7
    res = pf.fuzzy_join(
        la, rb, left_on="name", right_on="name",
        algorithms=["jaro_winkler", "trigram_jaccard"],
        weights=[0.6, 0.4], threshold=threshold,
    )
    predicted = set(zip(res["rec_id_a"].to_list(), res["rec_id_b"].to_list()))

    # Restrict ground truth to the subsample.
    a_ids = set(la["rec_id_a"].to_list())
    b_ids = set(rb["rec_id_b"].to_list())
    true_sub = {(x, y) for (x, y) in true if x in a_ids and y in b_ids}

    tp = len(predicted & true_sub)
    fp = len(predicted - true_sub)
    fn = len(true_sub - predicted)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print(f"subsample: {n} x {n} = {n*n:,} candidate pairs")
    print(f"true links in subsample : {len(true_sub):,}")
    print(f"predicted (thr>={threshold}): {len(predicted):,}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  precision={prec:.4f}  recall={rec:.4f}  F1={f1:.4f}")

    # Quick threshold sweep on the same subsample for the secondary goal.
    print("\n  threshold sweep (same n=1000, weighted_avg jw+trigram):")
    print(f"  {'thr':>5} {'pred':>8} {'prec':>8} {'rec':>8} {'f1':>8}")
    # Build once at thr=0, then sweep in Polars for speed.
    allpairs = pf.fuzzy_join(
        la, rb, left_on="name", right_on="name",
        algorithms=["jaro_winkler", "trigram_jaccard"],
        weights=[0.6, 0.4], threshold=0.0,
    )
    scored = list(zip(
        allpairs["rec_id_a"].to_list(),
        allpairs["rec_id_b"].to_list(),
        allpairs["score"].to_list(),
    ))
    for thr in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        pred = {(x, y) for (x, y, s) in scored if s >= thr}
        tp = len(pred & true_sub)
        fp = len(pred - true_sub)
        fn = len(true_sub - pred)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        print(f"  {thr:>5.2f} {len(pred):>8,} {p:>8.4f} {r:>8.4f} {f:>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("loading FEBRL4 ...")
    a, b, true = load_febrl()
    print(f"frame A: {a.shape}  frame B: {b.shape}  true links: {len(true):,}")

    # Fixed pair sample for metric-level benchmarks: 50k cross-sample pairs.
    pl.Config.set_tbl_rows(0)
    sample_n = 50_000
    pairs = (
        a.join(b, how="cross")
        .head(sample_n)
        .rename({"name": "a", "name_right": "b"})
        .select("a", "b")
    )
    print(f"pair sample: {pairs.height:,}")

    bench_single_metrics(pairs)
    bench_hybrid_vs_combine(pairs)
    bench_fuzzy_join_scaling(a, b)
    bench_deduplicate(a)
    bench_accuracy(a, b, true)


if __name__ == "__main__":
    main()
