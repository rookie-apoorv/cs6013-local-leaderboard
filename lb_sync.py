"""lb_sync.py - pull the latest local-leaderboard registries and refresh the page data.

    pip install huggingface_hub pandas pyarrow                  # once
    python lb_sync.py --repo <user>/cs6013-local-leaderboard   # first time (repo remembered)
    python lb_sync.py                 # sync, then open leaderboard.html as a file
    python lb_sync.py --serve         # sync, then serve the page on http://127.0.0.1:8765
                                      # (needed for the trace viewer; Ctrl+C to stop)
    python lb_sync.py --serve --no-sync   # serve the last synced data without syncing

Token: uses the HF login on this machine (`huggingface-cli login`), or --token, or the
HF_TOKEN environment variable. The repo is only ever read.

What a sync downloads: the question + submission registries, and from every results shard
only the light columns (scores, tokens, flags, extracted answer) - never the traces.
Unchanged shards are not fetched again. Traces are fetched only in --serve mode, per
question, when you open it in the page (each needed shard is downloaded once and cached).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
CACHE = HERE / "lb_local"
CONF = HERE / "lb_sync_config.json"
SETS_FILE = HERE / "lb_sets.json"        # saved question sets (written by the page via --serve)
LIGHT_SCHEMA = 2
LIGHT = ["submission_id", "uid", "seed", "config_id", "correct_grader", "parsed",
         "truncated", "n_completion_tokens", "score_error", "wait_per_1k", "looping",
         "pred_from_thinking", "grader_pred"]
TRACE_COLS = ["submission_id", "uid", "seed", "config_id", "completion", "finish_reason",
              "n_completion_tokens", "grader_pred", "correct_grader", "truncated", "parsed"]
Q_COLS = ["uid", "module", "source", "year", "topic", "level", "active", "answer_version",
          "problem", "answer"]
S_COLS = ["submission_id", "alias", "kind", "hf_id", "hf_revision", "week", "target",
          "submission_no", "size_frac", "method", "notes", "official_score", "registered_at"]


def log(*a):
    print(*a, flush=True)


def local_light(p):
    return CACHE / "results_light" / p.replace("/", "__")


def sync(repo, token, full=False):
    import pandas as pd
    from huggingface_hub import HfApi, HfFileSystem, hf_hub_download

    api = HfApi(token=token)
    t0 = time.time()
    log(f"syncing {repo} ...")
    tree = {f.path: getattr(f, "blob_id", None)
            for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
            if hasattr(f, "size")}  # files only
    (CACHE / "registry").mkdir(parents=True, exist_ok=True)
    (CACHE / "results_light").mkdir(parents=True, exist_ok=True)

    def get_file(path):
        return hf_hub_download(repo, path, repo_type="dataset", token=token,
                               local_dir=str(CACHE / "hub"), force_download=True)

    for name in ("questions/questions.parquet", "submissions/submissions.parquet"):
        if name not in tree:
            sys.exit(f"{name} not found in {repo} - run lb_import_base first")
        pd.read_parquet(get_file(name)).to_parquet(CACHE / "registry" / Path(name).name,
                                                   index=False)
    Q = pd.read_parquet(CACHE / "registry" / "questions.parquet")
    S = pd.read_parquet(CACHE / "registry" / "submissions.parquet")

    man_path = CACHE / "manifest.json"
    manifest = {} if (full or not man_path.exists()) else json.loads(man_path.read_text())
    if manifest.get("_schema") != LIGHT_SCHEMA:      # new light columns -> refetch all
        manifest = {}
    shards = sorted(p for p in tree if p.startswith("results/") and p.endswith(".parquet"))
    fs = HfFileSystem(token=token)
    fetched = 0
    for p in shards:
        if manifest.get(p) == tree[p] and local_light(p).exists():
            continue
        try:
            with fs.open(f"datasets/{repo}/{p}", "rb") as fh:
                df = pd.read_parquet(fh, columns=LIGHT)
        except Exception:  # fall back to a full download of this shard
            df = pd.read_parquet(get_file(p))
            df = df[[c for c in LIGHT if c in df.columns]]
        df.to_parquet(local_light(p), index=False)
        manifest[p] = tree[p]
        fetched += 1
        log(f"  fetched {p} ({len(df)} rows)")
    for p in [k for k in manifest if not k.startswith("_")]:
        if p not in tree:                             # shard deleted on the Hub
            local_light(p).unlink(missing_ok=True)
            manifest.pop(p)
    manifest["_schema"] = LIGHT_SCHEMA
    man_path.write_text(json.dumps(manifest, indent=1))

    paths = [p for p in manifest if not p.startswith("_")]
    parts = []
    for i, p in enumerate(paths):
        df = pd.read_parquet(local_light(p))
        df["_f"] = i
        parts.append(df)
    R = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=LIGHT + ["_f"])
    for c in LIGHT:
        if c not in R.columns:
            R[c] = None
    R = R.drop_duplicates(["submission_id", "uid", "seed", "config_id"])
    R = R[R.score_error.isna()]

    def col(df, c):
        if c not in df.columns:
            return [None] * len(df)
        out = []
        for v in df[c].tolist():
            if v is None or (isinstance(v, float) and v != v):
                out.append(None)
            else:
                out.append(v.item() if hasattr(v, "item") else v)
        return out

    q_index = {u: i for i, u in enumerate(Q.uid)}
    s_index = {s: i for i, s in enumerate(S.submission_id)}
    R = R[R.uid.isin(q_index) & R.submission_id.isin(s_index)]
    cfgs = sorted(set(R.config_id))
    c_index = {c: i for i, c in enumerate(cfgs)}

    def b(x):
        return int(bool(x)) if x == x and x is not None else 0

    data = {
        "repo": repo,
        "synced_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "questions": {c: col(Q, c) for c in Q_COLS},
        "submissions": {c: col(S, c) for c in S_COLS},
        "configs": cfgs,
        "shards": paths,
        "results": {
            "s": [s_index[x] for x in R.submission_id],
            "q": [q_index[x] for x in R.uid],
            "seed": [int(x) for x in R.seed],
            "c": [c_index[x] for x in R.config_id],
            "f": [int(x) for x in R["_f"]],
            "ok": [b(x) for x in R.correct_grader],
            "parsed": [b(x) for x in R.parsed],
            "trunc": [b(x) for x in R.truncated],
            "tok": [int(x) if x == x and x is not None else 0 for x in R.n_completion_tokens],
            "wait": [round(float(x), 2) if x == x and x is not None else None
                     for x in R.wait_per_1k],
            "loop": [b(x) for x in R.looping],
            "fromthink": [b(x) for x in R.pred_from_thinking],
            "pred": [None if x is None or x != x else str(x)[:60] for x in R.grader_pred],
        },
    }
    out = write_data(data, HERE)
    log(f"done in {time.time() - t0:.0f}s: {len(Q)} questions, {len(S)} submissions, "
        f"{len(R)} result rows ({fetched} shard(s) fetched) -> {out.name} "
        f"({out.stat().st_size / 1e6:.1f} MB)")
    return data


def write_data(data, folder):
    out = Path(folder) / "lb_data.js"
    out.write_text("window.LB_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# --export: a self-contained static site (index.html + lb_data.js)
# --public: privacy pass for a public site
# ---------------------------------------------------------------------------
PUBLIC_Q_COLS = ["uid", "module", "source", "year", "topic", "level", "active", "answer_version"]
PUBLIC_S_COLS = ["alias", "kind", "week", "target", "submission_no", "size_frac", "method",
                 "official_score"]


def make_public(data):
    """Aliases only (HF ids contain the roll number), no repo name, no problem text,
    no gold or extracted answers, no shard paths. Scores, tokens and flags are kept."""
    S = data["submissions"]
    n = len(S["submission_id"])
    ids, seen = [], set()
    for i in range(n):
        base = S["alias"][i] or (("base" if S["kind"][i] == "base" else "model") + f"-{i + 1}")
        pid, j = base, 2
        while pid in seen:
            pid, j = f"{base}-{j}", j + 1
        seen.add(pid)
        ids.append(pid)
    subs = {c: S.get(c, [None] * n) for c in PUBLIC_S_COLS}
    subs["submission_id"] = ids
    subs["alias"] = ids
    res = {k: v for k, v in data["results"].items() if k not in ("f", "pred")}
    return {
        "public": True,
        "repo": None,
        "synced_at": data["synced_at"],
        "questions": {c: data["questions"][c] for c in PUBLIC_Q_COLS if c in data["questions"]},
        "submissions": subs,
        "configs": data["configs"],
        "shards": [],
        "results": res,
    }


def export_site(data, folder, public):
    import shutil
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    page = HERE / "leaderboard.html"
    if not page.exists():
        sys.exit("leaderboard.html must sit next to lb_sync.py")
    shutil.copy(page, folder / "index.html")
    (folder / ".nojekyll").write_text("")              # serve files as-is on GitHub Pages
    out = write_data(make_public(data) if public else data, folder)
    log(f"exported {'public ' if public else ''}site -> {folder} "
        f"(index.html + {out.name}, {out.stat().st_size / 1e6:.2f} MB)")


# ---------------------------------------------------------------------------
# --serve: static files + /api/traces (fetches the needed shard once, then cached)
# ---------------------------------------------------------------------------
def serve(repo, token, port):
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    man_path = CACHE / "manifest.json"
    full_dir = CACHE / "full"
    full_man_path = CACHE / "full_manifest.json"
    lock = threading.Lock()

    def full_shard(path):
        manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
        if path.startswith("_") or path not in manifest:
            raise KeyError(f"unknown shard {path}")
        with lock:
            fm = json.loads(full_man_path.read_text()) if full_man_path.exists() else {}
            local = full_dir / path
            if fm.get(path) != manifest[path] or not local.exists():
                log(f"  downloading traces shard {path} ...")
                hf_hub_download(repo, path, repo_type="dataset", token=token,
                                local_dir=str(full_dir), force_download=True)
                fm[path] = manifest[path]
                full_man_path.write_text(json.dumps(fm, indent=1))
        return local

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(HERE), **k)

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/sets":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > 5_000_000:
                return self._json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body.get("sets"), list):
                    raise ValueError("expected {\"sets\": [...]}")
                tmp = SETS_FILE.with_suffix(".tmp")
                tmp.write_text(json.dumps({"sets": body["sets"]}, indent=1), encoding="utf-8")
                tmp.replace(SETS_FILE)               # atomic on the same disk
                return self._json(200, {"ok": True, "count": len(body["sets"])})
            except Exception as exc:
                return self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/api/ping":
                return self._json(200, {"ok": True})
            if u.path == "/api/sets":
                data = json.loads(SETS_FILE.read_text(encoding="utf-8")) if SETS_FILE.exists() \
                    else {"sets": []}
                return self._json(200, data)
            if u.path != "/api/traces":
                if u.path.startswith("/lb_local") or u.path.endswith(".json"):
                    return self._json(403, {"error": "not served"})
                return super().do_GET()
            qs = parse_qs(u.query)
            uid = (qs.get("uid") or [""])[0]
            rows = []
            try:
                for path in qs.get("shard", []):
                    t = pq.read_table(full_shard(path), columns=TRACE_COLS,
                                      filters=[("uid", "=", uid)])
                    rows += t.to_pylist()
                return self._json(200, {"uid": uid, "rows": rows})
            except Exception as exc:
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/leaderboard.html"
    log(f"serving {url}  (Ctrl+C to stop)")
    return srv, url


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", help="leaderboard dataset repo, e.g. user/cs6013-local-leaderboard")
    ap.add_argument("--token", help="HF token (default: local login / HF_TOKEN)")
    ap.add_argument("--serve", action="store_true", help="serve the page + trace viewer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-sync", action="store_true", help="skip syncing (with --serve)")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    ap.add_argument("--full", action="store_true", help="ignore the cache, re-read every shard")
    ap.add_argument("--export", metavar="DIR", help="also write a static site (index.html + data) to DIR")
    ap.add_argument("--public", action="store_true",
                    help="with --export: privacy pass (aliases only, no ids / repo / problem text / answers)")
    args = ap.parse_args()

    conf = json.loads(CONF.read_text()) if CONF.exists() else {}
    repo = args.repo or conf.get("repo")
    if not repo:
        sys.exit("first run: pass --repo <user>/cs6013-local-leaderboard")
    CONF.write_text(json.dumps({"repo": repo}, indent=2))
    token = args.token or os.environ.get("HF_TOKEN") or None

    data = None
    if not args.no_sync:
        data = sync(repo, token, full=args.full)
    if args.export:
        if data is None:
            txt = (HERE / "lb_data.js").read_text(encoding="utf-8")
            data = json.loads(txt[txt.index("=") + 1:].strip().rstrip(";"))
        export_site(data, args.export, args.public)
        return
    if args.serve:
        srv, url = serve(repo, token, args.port)
        if not args.no_open:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            log("stopped")
    elif not args.no_open and (HERE / "leaderboard.html").exists():
        webbrowser.open((HERE / "leaderboard.html").as_uri())


if __name__ == "__main__":
    main()
