#!/usr/bin/env python3
"""Compare sparkinfer vs a running llama.cpp server on the same token sequence.

  accuracy_compare.py <spark_score.txt> <tokenizer.json> <text> [server_url] [topk]
                      [--tail N] [--metric-label NAME]

Reads sparkinfer's teacher-forced score dump (from qwen3_gguf_score) and queries the
llama.cpp server (/completion, n_probs + cache_prompt) for the same per-position
distributions, then reports:
  - top-1 token agreement   (argmax_spark == argmax_llama)   -> implementation correctness
  - mean KL(llama || spark) over the top-k union             -> distribution closeness
  - perplexity for each engine (exp(-mean log p(actual next)))

--tail N scores only the LAST N positions of the stream. This is what makes the long-context
probe affordable: llama is queried once per scored position, so scoring all ~8k positions of a
long stream would mean ~8k HTTP round-trips. A long prefix with a short scored tail costs N calls
(cache_prompt keeps the prefix warm) while every scored position sits at a seqlen past the
int8-MMA and sparse-KV engagement thresholds — which is the whole point of the probe.

--metric-label NAME renames the machine-readable METRIC line (e.g. METRIC_LONG), so accuracy.sh
can emit several passes and still hand evaluate.sh exactly one unambiguous `METRIC ` line.
"""
import sys, json, math, urllib.request
from tokenizers import Tokenizer

argv = sys.argv[1:]
TAIL = 0
LABEL = "METRIC"
if "--tail" in argv:
    i = argv.index("--tail"); TAIL = int(argv[i + 1]); del argv[i:i + 2]
if "--metric-label" in argv:
    i = argv.index("--metric-label"); LABEL = argv[i + 1]; del argv[i:i + 2]

score_path, tok_path, text_path = argv[0], argv[1], argv[2]
URL  = argv[3] if len(argv) > 3 else "http://localhost:8081"
# llama top-k to query per position. MUST be <= sparkinfer's dump depth (accuracy.sh dumps 128) so
# every token llama gives mass to is present in sparkinfer's distribution — otherwise it gets FLOOR'd
# and KL is inflated by a truncation artifact (the metric bug that read 0.14-0.33 instead of ~0.02).
TOPK = int(argv[4]) if len(argv) > 4 else 64
FLOOR = -20.0

# 3rd arg is either a file of space-separated token ids (the EXACT prompt scored — produced by
# gen_eval_prompt.py so sparkinfer and llama see the identical sequence) or, legacy, plain text.
_raw = open(text_path).read().strip()
_toks = _raw.split()
if _toks and all(t.lstrip("-").isdigit() for t in _toks):
    ids = [int(t) for t in _toks]
else:
    ids = Tokenizer.from_file(tok_path).encode(_raw).ids

def llama_dist(prefix):
    req = {"prompt": prefix, "n_predict": 1, "n_probs": TOPK, "temperature": 0, "cache_prompt": True}
    r = urllib.request.urlopen(urllib.request.Request(
        URL + "/completion", data=json.dumps(req).encode(),
        headers={"Content-Type": "application/json"}), timeout=120)
    tl = json.load(r)["completion_probabilities"][0]["top_logprobs"]
    return {e["id"]: e["logprob"] for e in tl}

spark = {}
for line in open(score_path):
    if not line.startswith("S "): continue
    p = line.split(); i = int(p[1][2:]); am = int(p[3][3:]); lp = float(p[4][3:])
    top = {int(x.split(":")[0]): float(x.split(":")[1]) for x in line.split("top=", 1)[1].split(",")}
    spark[i] = {"am": am, "lp": lp, "top": top}

match = n = 0; snll = lnll = 0.0; klsum = 0.0
lo = max(0, (len(ids) - 1) - TAIL) if TAIL > 0 else 0
for i in range(lo, len(ids) - 1):
    if i not in spark: continue
    ld = llama_dist(ids[:i + 1]); lam = max(ld, key=ld.get); n += 1
    if spark[i]["am"] == lam: match += 1
    snll += -spark[i]["lp"]; lnll += -ld.get(ids[i + 1], FLOOR)
    sd = spark[i]["top"]; U = set(ld) | set(sd)
    P = {k: math.exp(ld.get(k, FLOOR)) for k in U}; Q = {k: math.exp(sd.get(k, FLOOR)) for k in U}
    ps = sum(P.values()); qs = sum(Q.values()); kl = 0.0
    for k in U:
        pp = P[k] / ps; qq = Q[k] / qs
        if pp > 0: kl += pp * math.log(pp / max(qq, 1e-12))
    klsum += kl

if n == 0:
    print(f"{LABEL} top1=0 kl=99 ppl_spark=0 ppl_llama=0   (NO SCORED POSITIONS)")
    sys.exit(1)
print(f"positions             : {n}" + (f"  (tail {TAIL} of {len(ids)})" if TAIL else ""))
print(f"scored range          : {lo}..{len(ids) - 2}  (seqlen {lo + 1}..{len(ids) - 1})")
print(f"token-match (top-1)   : {match}/{n} = {match/n:.3f}   (bar >= 0.90)")
print(f"mean KL(llama||spark) : {klsum/n:.4f} nats  (top-k approx)")
print(f"PPL sparkinfer        : {math.exp(snll/n):.3f}  (exact, full softmax)")
print(f"PPL llama.cpp         : {math.exp(lnll/n):.3f}  (top-{TOPK}+floor; inflated)")
# unambiguous machine-readable line for evaluate.sh (avoid parsing the human text above)
print(f"{LABEL} top1={match/n:.6f} kl={klsum/n:.6f} ppl_spark={math.exp(snll/n):.4f} ppl_llama={math.exp(lnll/n):.4f}")
