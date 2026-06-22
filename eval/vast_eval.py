#!/usr/bin/env python3
"""Automatic evaluation on a vast.ai GPU: provision (or reuse) → build/correctness/speed → label → teardown.

Requires VAST_API_KEY (`vastai set api-key <key>`). The numeric label is computed on-box by
bench/scripts/label.py (deterministic) — this script only orchestrates.

  # reuse an existing box (started if stopped, STOPPED again after the eval — the default):
  python eval/vast_eval.py --reuse 42134865 --frontier 164 --ceiling 366 --ref main

  # evaluate then DESTROY (frees the disk too), or --keep to leave it running:
  python eval/vast_eval.py --ref <git-ref> --frontier 164 --ceiling 366 --destroy

By default the instance is STOPPED after every eval: compute billing pauses while the disk and
cached weights (/workspace/models) persist for a fast next run. --keep leaves it running.

Env: VAST_API_KEY, SSH_KEY (default ~/.ssh/id_ed25519), LLAMACPP_DIR, EVAL_IMAGE, EVAL_REPO.
"""
import argparse, json, os, subprocess, sys, time
from vastai import VastAI

REPO    = os.environ.get("EVAL_REPO",  "https://github.com/gittensor-ai-lab/sparkinfer")
IMAGE   = os.environ.get("EVAL_IMAGE", "nvidia/cuda:12.8.0-devel-ubuntu24.04")   # needs nvcc for sm_120
SSH_KEY = os.path.expanduser(os.environ.get("SSH_KEY", "~/.ssh/id_ed25519"))
LLAMACPP_DIR = os.environ.get("LLAMACPP_DIR", "/workspace/.llamacpp")            # persists across stop/start

def sh(host, port, cmd, timeout=3600):
    return subprocess.run(
        ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
         "-p", str(port), f"root@{host}", cmd], capture_output=True, text=True, timeout=timeout)

def info_of(v, iid):
    return next((i for i in v.show_instances() if i.get("id") == iid), None)

def endpoint(info):
    """Prefer the DIRECT endpoint (public_ipaddr + mapped :22) — the vast SSH proxy authenticates
    against account keys and is flakier; the direct port uses the instance's authorized_keys."""
    ip = info.get("public_ipaddr"); ports = info.get("ports") or {}
    m = ports.get("22/tcp")
    if ip and m:
        return ip.strip(), int(m[0]["HostPort"])
    return info.get("ssh_host"), int(info.get("ssh_port"))

def wait_ssh(host, port, tries=60):
    for _ in range(tries):
        try:
            if sh(host, port, "echo ok", timeout=15).stdout.strip().endswith("ok"): return True
        except Exception: pass
        time.sleep(10)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="main")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling",  type=float, default=0)
    ap.add_argument("--reuse", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="leave the instance running after eval (default: stop it)")
    ap.add_argument("--destroy", action="store_true", help="destroy after eval instead of stopping (also frees the disk)")
    ap.add_argument("--gpu", default="RTX_5090")
    ap.add_argument("--image", default=IMAGE)
    args = ap.parse_args()

    v = VastAI(); created = False; iid = args.reuse
    if not iid:
        offers = v.search_offers(query=f"gpu_name={args.gpu} num_gpus=1 cuda_vers>=12.8 inet_down>=100",
                                 order="dph_total", limit=10)
        if not offers: sys.exit("no matching offers")
        off = offers[0]
        print(f">> create on offer {off['id']} {off.get('gpu_name')} ${off.get('dph_total'):.3f}/hr")
        inst = v.create_instance(id=off["id"], image=args.image, disk=120, ssh=True, direct=True)
        iid = inst.get("new_contract") or inst.get("id"); created = True

    info = info_of(v, iid)
    if info and info.get("actual_status") != "running":
        print(f">> starting instance {iid} ...")
        try: v.start_instance(id=iid)
        except Exception as e: print("start:", str(e)[:150])
    for _ in range(60):
        info = info_of(v, iid)
        if info and info.get("actual_status") == "running" and (info.get("public_ipaddr") or info.get("ssh_host")): break
        time.sleep(10)
    if not info: sys.exit(f"instance {iid} not found")
    host, port = endpoint(info)
    print(f">> instance {iid}: ssh root@{host}:{port}")
    if not wait_ssh(host, port): sys.exit("ssh never came up")

    try:
        setup = ("export DEBIAN_FRONTEND=noninteractive; "
                 "command -v git >/dev/null || (apt-get update -q && apt-get install -y -q git curl cmake build-essential); "
                 "pip install -q --break-system-packages huggingface_hub tokenizers >/dev/null 2>&1 || true; "
                 f"if [ -d /root/sparkinfer/.git ]; then cd /root/sparkinfer && git fetch -q origin && git checkout -q {args.ref} && git pull -q origin {args.ref}; "
                 f"else git clone -q {REPO} /root/sparkinfer && cd /root/sparkinfer && git checkout -q {args.ref}; fi")
        if sh(host, port, setup, timeout=1800).returncode: print(">> setup warnings (continuing)")
        ev = (f"cd /root/sparkinfer && MODELS_DIR=/workspace/models LLAMACPP_DIR={LLAMACPP_DIR} "
              f"bench/scripts/evaluate.sh --ref {args.ref} --frontier {args.frontier} --ceiling {args.ceiling}")
        r = sh(host, port, ev, timeout=10800)
        sys.stdout.write(r.stdout[-4000:])
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if line:
            print("\n=== VERDICT ==="); print(json.dumps(json.loads(line[len("RESULT_JSON "):]), indent=2))
        else:
            print("\n!! no RESULT_JSON; stderr tail:\n" + r.stderr[-1500:])
    finally:
        # Default: STOP after every eval — pauses compute billing, keeps disk + weights for fast reuse.
        if args.keep:
            print(f">> leaving instance {iid} running (--keep)")
        elif args.destroy:
            print(f">> destroying instance {iid} (disk freed)");
            try: v.destroy_instance(id=iid)
            except Exception as e: print("destroy:", str(e)[:150])
        else:
            print(f">> stopping instance {iid} — disk/weights persist; resume with --reuse {iid}")
            try: v.stop_instance(id=iid)
            except Exception as e: print("stop:", str(e)[:150])

if __name__ == "__main__":
    main()
