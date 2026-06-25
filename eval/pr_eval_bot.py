#!/usr/bin/env python3
"""sparkinfer PR auto-evaluator (bot).

Polls open PRs; for any PR whose head commit hasn't been evaluated yet, runs the vast.ai
evaluation (build → correctness → speed → label), applies an `eval:<LABEL>` label, and posts the
result as a PR comment. **Never merges** — merging is manual after review.

Designed to run on a 30-min schedule (system cron or a Claude agent). Idempotent: a commit is
evaluated once (tracked by a hidden marker in the bot's comment), so re-runs only pick up new
commits and only spin the GPU when there's new work.

  python eval/pr_eval_bot.py --instance 42134865 --frontier 164 --ceiling 366

Needs: `gh` authenticated, VAST_API_KEY saved (vastai), and the eval:* labels (eval/setup_labels.sh).
"""
import argparse, datetime, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# vast_eval.py self-heals dead boxes (recreates them) and writes the working instance id here;
# prefer it over --instance so we reuse the recreated box instead of retrying the dead one.
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))
def current_instance(default):
    try: return int(open(INSTANCE_FILE).read().strip())
    except Exception: return default

# Subsystem buckets for the deterministic area:<name> label (from a PR's top-level changed
# dirs — no AI). Categorization/display only: SN74 scoring is speedup-only (the eval:* tier),
# NOT a per-subsystem budget.
AREAS = {"kernels", "runtime", "moe", "bench"}

# RTX 5090 evaluation is OPT-IN: a maintainer reviews a PR and adds EVAL_GATE_LABEL to greenlight it
# for the (expensive) on-device eval. PRs without it get NOT_TESTED_LABEL and are NOT evaluated — this
# stops the bot burning GPU on unvetted / spam / gaming PRs before a human has looked at them.
EVAL_GATE_LABEL = "test-on-5090"
NOT_TESTED_LABEL = "not-tested"

def gh(args):
    return subprocess.run(["gh"] + args, capture_output=True, text=True)

# ---- contributor denylist (eval-gaming / sybil block) ----
# .github/blocked-contributors.txt lists GitHub logins (one per line, # = comment). A PR is blocked
# if its opener OR any commit author/committer is listed. Blocked PRs are labeled, commented, closed,
# and NOT evaluated. Evidence per account: .github/FLAGGED.md
DENYLIST_FILE = os.path.join(ROOT, ".github", "blocked-contributors.txt")
FLAG_LABEL = "flagged:gaming"

def load_denylist():
    try:
        out = set()
        for line in open(DENYLIST_FILE):
            s = line.split("#", 1)[0].strip().lower()
            if s: out.add(s)
        return out
    except Exception:
        return set()

def pr_involved_logins(repo, num):
    """Every GitHub login tied to a PR: opener + each commit's author and committer."""
    owner, r = _owner_repo(repo)
    logins = set()
    info = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "author"]).stdout or "{}")
    if info.get("author", {}).get("login"): logins.add(info["author"]["login"].lower())
    out = subprocess.run(["gh", "api", f"repos/{owner}/{r}/pulls/{num}/commits",
                          "--jq", ".[] | (.author.login // \"\") + \"\\n\" + (.committer.login // \"\")"],
                         capture_output=True, text=True)
    for l in (out.stdout or "").splitlines():
        if l.strip(): logins.add(l.strip().lower())
    return logins

def close_blocked_pr(repo, num, hits):
    """Label flagged:gaming, comment with the reason, and close the PR. Returns True on close."""
    add_label(repo, num, FLAG_LABEL)
    who = ", ".join(f"`{h}`" for h in sorted(hits))
    body = ("<!-- sparkinfer-flagged -->\n"
            "## 🚩 Flagged: eval-gaming\n\n"
            f"This PR involves an account blocked for gaming the SN74 emission mechanism "
            f"(sybil / coordinated duplicate farming): {who}.\n\n"
            "Per the project's no-gaming policy these accounts are blocked: the PR is **not "
            "evaluated, scored, or merged**. See [`.github/FLAGGED.md`]"
            "(../blob/main/.github/FLAGGED.md) for the evidence and record.")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])
    return gh(["pr", "close", str(num), "-R", repo]).returncode == 0

def block_account(login, reason):
    """Append a login to the denylist file + a reason to FLAGGED.md (deduped)."""
    cur = load_denylist()
    if login.lower() not in cur:
        with open(DENYLIST_FILE, "a") as f: f.write(f"\n{login}\n")
    with open(FLAG_FILE, "a") as f:
        f.write(f"\n## {datetime.date.today().isoformat()} — `{login}` (auto-blocked)\n\n{reason}\n")

# ---- copycat detection (a later PR that re-submits an earlier PR's diff) ----
# A PR is a copycat if its added lines are largely contained in an EARLIER PR touching the same
# file(s). Copycats are labeled `copycat`, commented (citing the original), and NOT evaluated.
# Logged to .github/copycats.json; a 2nd copycat by the same author auto-adds them to the denylist.
FLAG_FILE = os.path.join(ROOT, ".github", "FLAGGED.md")
COPYCAT_LABEL = "copycat"
COPYCAT_LOG = os.path.join(ROOT, ".github", "copycats.json")
COPYCAT_CONTAINMENT = 0.80   # ≥80% of the copy's added lines also appear in the original
COPYCAT_STRIKES = 2          # this many copycats by one author -> denylist

def pr_fingerprint(repo, num):
    """(changed files, normalized non-empty added lines) from the PR's unified diff."""
    diff = gh(["pr", "diff", str(num), "-R", repo]).stdout or ""
    files, added = set(), set()
    for line in diff.splitlines():
        if line.startswith("+++ ") or line.startswith("--- "):
            p = line[4:].strip()
            if p.startswith(("a/", "b/")): p = p[2:]
            if p and p != "/dev/null": files.add(p)
        elif line.startswith("+") and not line.startswith("+++"):
            s = line[1:].strip()
            if s and not s.startswith(("//", "#", "/*", "*")): added.add(s)  # skip comment-only lines
    return files, added

def containment(copy_added, orig_added):
    if not copy_added: return 0.0
    return len(copy_added & orig_added) / len(copy_added)

def load_copycat_log():
    try: return json.load(open(COPYCAT_LOG))
    except Exception: return []

def save_copycat_log(log):
    os.makedirs(os.path.dirname(COPYCAT_LOG), exist_ok=True)
    with open(COPYCAT_LOG, "w") as f: json.dump(log, f, indent=2)

def push_github_state(msg):
    subprocess.run(["git", "-C", ROOT, "add", ".github/copycats.json",
                    ".github/blocked-contributors.txt", ".github/FLAGGED.md"], capture_output=True)
    if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet"]).returncode == 0:
        return
    subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", msg], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "pull", "-q", "--rebase", "origin", "main"], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "push", "-q", "origin", "main"], capture_output=True)

def flag_copycat(repo, num, original, author):
    add_label(repo, num, COPYCAT_LABEL)
    body = (f"<!-- sparkinfer-copycat -->\n## 🐈 Flagged: copycat\n\n"
            f"This PR re-submits substantially the same diff as the earlier #{original}. "
            f"Duplicating an existing PR's work does not earn a score — copycats are **not "
            f"evaluated or merged**.\n\nRepeated copycatting (≥{COPYCAT_STRIKES}) results in an "
            f"automatic block. See [`.github/COPYCATS.md`](../blob/main/.github/COPYCATS.md).")
    gh(["pr", "comment", str(num), "-R", repo, "--body", body])

def evaluated_commits(repo, num):
    r = gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        for m in re.finditer(r"<!-- sparkinfer-eval:([0-9a-f]+) -->", c.get("body", "")):
            done.add(m.group(1))
    return done

def areas_for_pr(repo, num):
    """Subsystems a PR touches, from its changed file paths (deterministic, no AI)."""
    files = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "files"]).stdout or "{}").get("files", [])
    return {f["path"].split("/", 1)[0] for f in files} & set(AREAS)

def pr_checkbox_tested(repo, num):
    """True if the PR body has the template's 'Tested on RTX 5090' checkbox ticked
    (a markdown '[x]' on a line mentioning 5090). Self-attested greenlight for eval."""
    body = (json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "body"]).stdout or "{}")
            .get("body") or "")
    return any(re.search(r"\[\s*[xX]\s*\]", ln) and "5090" in ln for ln in body.splitlines())

def _owner_repo(repo):
    parts = repo.split("/"); return parts[0], parts[1]

def labels_on(repo, num):
    owner, r = _owner_repo(repo)
    out = subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels",
                          "--jq", "[.[].name]"], capture_output=True, text=True)
    try: return set(json.loads(out.stdout))
    except Exception: return set()

def add_label(repo, num, label):
    owner, r = _owner_repo(repo)
    subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels",
                    "--method", "POST", "-f", f"labels[]={label}"],
                   capture_output=True, text=True)

def remove_label(repo, num, label):
    owner, r = _owner_repo(repo)
    subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels/{label}",
                    "--method", "DELETE"], capture_output=True, text=True)

def apply_area_labels(repo, num, areas):
    want = {f"area:{a}" for a in areas}
    have = {l for l in labels_on(repo, num) if l.startswith("area:")}
    for lab in want - have: add_label(repo, num, lab)
    for lab in have - want: remove_label(repo, num, lab)

def render(res, oid):
    label = res.get("label", "?")
    icon = {"REJECT": "❌", "none": "⚪", "BASELINE": "📊"}.get(label, "✅")
    rows = [f"| **label** | `eval:{label}` |",
            f"| decode | {res.get('tps','?')} tok/s |",
            f"| correctness | top-1 {res.get('top1',0)*100:.1f}% · KL {res.get('kl','?')} |"]
    if "frontier_tps" in res and res["frontier_tps"]:
        rows.insert(2, f"| vs frontier | {res['frontier_tps']} tok/s → "
                       f"{res.get('pct_over_frontier', 0):+.1f}% ({res.get('delta_tps',0):+.1f}) |")
    note = {"REJECT": f"Failed the correctness gate: {res.get('reason','')}. Not a valid submission.",
            "none": "Within the significance gate — no *verified* speedup over the current frontier.",
            "BASELINE": "No frontier was set; this run establishes it."
            }.get(label, "Verified speedup over the live frontier.")
    return (f"<!-- sparkinfer-eval:{oid} -->\n"
            f"## {icon} sparkinfer auto-eval — `{oid}`\n\n"
            f"| metric | value |\n|---|---|\n" + "\n".join(rows) + "\n\n"
            f"{note}\n\n"
            f"_RTX 5090 (sm_120) · built from source · correctness vs llama.cpp. "
            f"Automated — **not merged**; merge manually after review._")

# ---- live dashboard: data.json is canonical; data.js is generated for the page ----
DASH = os.path.join(ROOT, "dashboard")
DATA_JSON = os.path.join(DASH, "data.json")
FRONTIER_LABELS = {"XL", "L", "M", "S", "XS", "BASELINE"}

def load_dash():
    try:
        with open(DATA_JSON) as f: return json.load(f)
    except Exception:
        return None

def write_dash(data):
    with open(DATA_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DASH, "data.js"), "w") as f:
        f.write("// Generated by eval/pr_eval_bot.py from data.json — do not edit by hand.\n")
        f.write("window.SPARKINFER = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n")

def push_dash(msg):
    subprocess.run(["git", "-C", ROOT, "add", "dashboard/data.json", "dashboard/data.js"], capture_output=True)
    if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet"]).returncode == 0:
        return  # nothing changed
    subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", msg], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "pull", "-q", "--rebase", "origin", "main"], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "push", "-q", "origin", "main"], capture_output=True)

def update_dashboard(repo, pr, areas, res):
    """Upsert the PR's verdict into the dashboard, ratchet the frontier, regenerate + push."""
    data = load_dash()
    if data is None: return
    num = pr["number"]
    entry = {"num": num, "title": pr.get("title", ""), "areas": sorted(areas),
             "label": res.get("label"), "tps": res.get("tps"),
             "delta_pct": res.get("pct_over_frontier"),
             "url": f"https://github.com/{repo}/pull/{num}"}
    data["prs"] = [p for p in data.get("prs", []) if p.get("num") != num]
    data["prs"].insert(0, entry)
    data["prs"] = data["prs"][:50]
    if (res.get("pass") and res.get("label") in FRONTIER_LABELS
            and (res.get("tps") or 0) > data["status"].get("frontier_tps", 0)):
        data["status"]["frontier_tps"] = res["tps"]            # ratchet the live frontier
        # Accuracy shown on the dashboard is the FRONTIER's accuracy — refresh it from the same
        # eval that just set the frontier, so token-match/KL never go stale (they used to be a
        # manual seed the bot never touched).
        if res.get("top1") is not None: data["status"]["token_match"] = round(res["top1"], 4)
        if res.get("kl") is not None:   data["status"]["kl"] = round(res["kl"], 4)
    data["updated"] = datetime.date.today().isoformat()
    write_dash(data)
    push_dash(f"dashboard: PR #{num} -> eval:{res.get('label')} ({res.get('tps')} tok/s)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True, help="vast.ai instance id to reuse")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling", type=float, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true", help="evaluate + print, but don't label/comment")
    args = ap.parse_args()

    dash = load_dash()
    frontier = dash["status"]["frontier_tps"] if dash else args.frontier   # live ledger frontier
    # OLDEST-FIRST: evaluate ascending by PR number so the original of any duplicate is seen before
    # its copy, and the earliest submitter is graded first (fairness + copycat attribution).
    prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "open",
                         "--json", "number,headRefName,headRefOid,title,isCrossRepository,labels"]).stdout or "[]")
    prs.sort(key=lambda p: p["number"])
    if not prs:
        print("no open PRs"); return

    # Fingerprint EVERY PR (open + closed + merged) so an open PR can be compared against any earlier
    # one — copies often target already-merged originals. Built once, ascending by number.
    all_prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "all",
                             "--json", "number,author", "-L", "300"]).stdout or "[]")
    all_nums = sorted(p["number"] for p in all_prs)
    pr_author = {p["number"]: (p.get("author") or {}).get("login", "?") for p in all_prs}
    fps = {n: pr_fingerprint(args.repo, n) for n in all_nums}
    copy_log = load_copycat_log()
    logged_copycats = {e["pr"] for e in copy_log}
    state_changed = False

    def find_original(num):
        """Earliest PR by a DIFFERENT author (shared file) whose added lines contain this PR's diff.
        Self-resubmissions (same author iterating on their own earlier PR) are NOT copycats."""
        files, added = fps.get(num, (set(), set()))
        if not added: return None
        me = pr_author.get(num, "?")
        for earlier in all_nums:
            if earlier >= num: break
            if pr_author.get(earlier, "?") == me: continue   # ignore one's own earlier PRs
            ef, ea = fps.get(earlier, (set(), set()))
            if (files & ef) and containment(added, ea) >= COPYCAT_CONTAINMENT:
                return earlier
        return None

    # Collect PRs that actually need evaluation before starting the GPU instance.
    denylist = load_denylist()
    pending = []
    for pr in prs:
        num, branch, oid = pr["number"], pr["headRefName"], pr["headRefOid"][:7]
        ref = f"pull/{num}/head" if pr.get("isCrossRepository") else branch
        # Gate 1 — blocked contributor: never spend GPU on a flagged/sybil PR.
        hits = pr_involved_logins(args.repo, num) & denylist
        if hits:
            print(f"PR #{num}: BLOCKED (denylisted: {', '.join(sorted(hits))}) — flag + close, no eval")
            if not args.dry_run: close_blocked_pr(args.repo, num, hits)
            continue
        # Gate 2 — copycat: re-submits a DIFFERENT author's earlier diff. Label, log, skip eval;
        # 2nd strike -> block. (Self-resubmissions are excluded by find_original.)
        original = find_original(num)
        if original is not None:
            author = pr_author.get(num, "?")
            print(f"PR #{num}: COPYCAT of #{original} by {pr_author.get(original,'?')} "
                  f"(author {author}) — flag, no eval")
            if not args.dry_run and num not in logged_copycats:
                flag_copycat(args.repo, num, original, author)
                copy_log.append({"pr": num, "author": author, "original": original,
                                 "date": datetime.date.today().isoformat()})
                logged_copycats.add(num); state_changed = True
                strikes = sum(1 for e in copy_log if e["author"] == author)
                if strikes >= COPYCAT_STRIKES and author.lower() not in load_denylist():
                    print(f"  -> {author} hit {strikes} copycats — auto-blocking")
                    block_account(author, f"Auto-blocked after {strikes} copycat PRs "
                                  f"(#{', #'.join(str(e['pr']) for e in copy_log if e['author']==author)}).")
                    close_blocked_pr(args.repo, num, {author})
            continue
        areas = areas_for_pr(args.repo, num)
        print(f"PR #{num} @ {oid}: areas={sorted(areas) or ['(none)']} ref={ref}")
        if not args.dry_run: apply_area_labels(args.repo, num, areas)
        if oid in evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {oid}: already evaluated — skip eval"); continue
        # Gate 3 — greenlight: a PR is evaluated only if the contributor checked the
        # "Tested on RTX 5090" box in the PR template, OR a maintainer added `test-on-5090`
        # (manual override). A checked box auto-applies the label. Otherwise -> `not-tested`,
        # skip (no GPU). Re-checking the box / adding the label triggers eval on the next poll.
        pr_labels = {l["name"] for l in pr.get("labels", [])}
        greenlit = (EVAL_GATE_LABEL in pr_labels) or pr_checkbox_tested(args.repo, num)
        if not greenlit:
            print(f"PR #{num}: not greenlit (RTX-5090 box unchecked, no {EVAL_GATE_LABEL}) "
                  f"— mark not-tested, skip eval")
            if not args.dry_run and NOT_TESTED_LABEL not in pr_labels:
                add_label(args.repo, num, NOT_TESTED_LABEL)
            continue
        print(f"PR #{num}: greenlit for RTX 5090 eval")
        if not args.dry_run:
            if EVAL_GATE_LABEL not in pr_labels: add_label(args.repo, num, EVAL_GATE_LABEL)
            if NOT_TESTED_LABEL in pr_labels: remove_label(args.repo, num, NOT_TESTED_LABEL)
        pending.append((pr, num, branch, oid, ref, areas))

    if not args.dry_run and state_changed:
        save_copycat_log(copy_log)
        push_github_state("eval: record copycat detections + any auto-blocks")

    if not pending:
        print("done — no merges (manual)."); return

    if args.dry_run:
        print("--- dry-run: would evaluate (oldest-first): " +
              ", ".join(f"#{n}" for _, n, *_ in pending)); return

    # Run all pending evals on the SAME instance: pass --keep so vast_eval.py never stops/destroys
    # the box mid-queue. The bot stops the instance once after ALL PRs finish (or if the instance
    # dies, subsequent PRs self-heal by provisioning a new one).
    for i, (pr, num, branch, oid, ref, areas) in enumerate(pending):
        # Re-read the frontier each iteration: a PR that passed earlier in THIS run may have
        # ratcheted it (update_dashboard writes data.json), and later PRs must be graded against
        # the new best — otherwise two PRs could both "beat" the same stale baseline.
        d = load_dash()
        cur_frontier = d["status"]["frontier_tps"] if d else frontier
        print(f"PR #{num} @ {oid}: evaluating '{ref}' (frontier={cur_frontier}) ...")
        r = subprocess.run([sys.executable, os.path.join(HERE, "vast_eval.py"),
                            "--reuse", str(current_instance(args.instance)), "--ref", ref,
                            "--frontier", str(cur_frontier), "--ceiling", str(args.ceiling),
                            "--keep"],          # keep instance alive — bot stops it after all PRs
                           cwd=ROOT, capture_output=True, text=True, timeout=14400)
        # If vast_eval self-healed to a new instance, track the new id for the next PR.
        for l in r.stdout.splitlines():
            if l.startswith("NEW_INSTANCE_ID "):
                try:
                    new_id = int(l.split()[1])
                    with open(INSTANCE_FILE, "w") as f: f.write(str(new_id))
                    print(f"  (instance updated to {new_id})")
                except Exception: pass
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if not line:
            log = (r.stdout + r.stderr)[-1500:]
            print(f"PR #{num}: eval produced no result\n{log}")
            body = (f"<!-- sparkinfer-eval:{oid} -->\n⚠️ **sparkinfer auto-eval errored** for `{oid}` "
                    f"— re-run manually.\n\n<details><summary>log tail</summary>\n\n```\n{log}\n```\n</details>")
            res, label = None, None
        else:
            res = json.loads(line[len("RESULT_JSON "):]); label = res["label"]; body = render(res, oid)
            print(f"PR #{num}: {json.dumps(res)}")
        if args.dry_run:
            print("--- dry-run, not posting ---\n" + body); continue
        if label:
            for lab in {l for l in labels_on(args.repo, num) if l.startswith("eval:")}:
                remove_label(args.repo, num, lab)
            add_label(args.repo, num, f"eval:{label}")
        gh(["pr", "comment", str(num), "-R", args.repo, "--body", body])
        print(f"PR #{num}: posted {'eval:'+label if label else 'error'} — NOT merged.")
        if res: update_dashboard(args.repo, pr, areas, res)

    # Stop (not destroy) the instance after all PRs — disk/model cache persists for next run.
    final_iid = current_instance(args.instance)
    if final_iid:
        print(f">> stopping instance {final_iid} — model cache persists for next run")
        subprocess.run(["vastai", "stop", "instance", str(final_iid)], capture_output=True)
    print("done — no merges (manual).")

if __name__ == "__main__":
    main()
