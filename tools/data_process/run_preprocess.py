"""Run independent first-stage dataset preprocessors concurrently in subprocesses.

Usage:
  python tools/data_process/run_preprocess.py --group train --limit 128 --jobs 10
  python tools/data_process/run_preprocess.py --group train --jobs 10                # Full dataset
  python tools/data_process/run_preprocess.py --only musique,qasper --limit 64

Groups:
  train -> first-stage training sets (10 preprocessors)
  test  -> first-stage evaluation sets
"""

import argparse
import concurrent.futures as cf
import os
import subprocess
import sys
import time

PRE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess")

# Each item is (script name, extra arguments); a script may appear with different arguments.
GROUPS = {
    "train": [(n, []) for n in [
        "musique", "twowiki", "narrativeqa", "longmit", "qasper",
        "agnews", "dbpedia", "longalpaca", "longalign", "ldc"]],
    # TTL evaluation uses MAB (mab_ttl).
    "test": [
        ("infbench", []), ("mab", []),
        ("longmemeval", []), ("hotpotqa", []), ("locomo", []),
        ("musique", ["--split", "test"]),
        ("twowiki", ["--split", "validation", "--out", "test"]),
    ],
}


def run_one(name: str, extra: list[str], limit: int | None) -> tuple[str, int, str]:
    cmd = [sys.executable, os.path.join(PRE, f"{name}.py"), *extra]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    label = name + (" " + " ".join(extra) if extra else "")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-4:])
    status = "OK" if p.returncode == 0 else f"FAIL(rc={p.returncode})"
    return label, p.returncode, f"[{status} {dt:.1f}s] {label}\n{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS), default=None)
    ap.add_argument("--only", default="", help="逗号分隔 name，覆盖 group")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    if args.only:
        items = [(s.strip(), []) for s in args.only.split(",") if s.strip()]
    elif args.group:
        items = GROUPS[args.group]
    else:
        ap.error("需指定 --group 或 --only")

    print(f"==> 并发跑 {len(items)} 个: {[n for n, _ in items]} (jobs={args.jobs}, limit={args.limit})")
    failed = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, n, extra, args.limit) for n, extra in items]
        for fut in cf.as_completed(futs):
            label, rc, msg = fut.result()
            print(msg, flush=True)
            if rc != 0:
                failed.append(label)
    print(f"\n==== 完成。成功 {len(items) - len(failed)} / {len(items)}，失败: {failed or '(无)'} ====")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
