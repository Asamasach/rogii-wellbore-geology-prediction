"""Install Python data stack on fteam6 — robust fallback chain."""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(
    os.environ["FTEAM6_HOST"],
    port=int(os.environ["FTEAM6_PORT"]),
    username=os.environ["FTEAM6_USER"],
    password=os.environ["FTEAM6_PASS"],
    timeout=15,
)

PASS = os.environ["FTEAM6_PASS"]


def run(c, t=600):
    print(f"$ {c[:200]}{'...' if len(c) > 200 else ''}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t, get_pty=False)
    for ln in iter(o.readline, ""):
        if not ln: break
        try: print(ln.rstrip(), flush=True)
        except UnicodeEncodeError: print(ln.encode("ascii", "replace").decode().rstrip(), flush=True)
    err = e.read().decode(errors="replace").rstrip()
    if err:
        try: print(f"[stderr] {err}", flush=True)
        except UnicodeEncodeError: print("[stderr]", err.encode("ascii", "replace").decode())
    rc = o.channel.recv_exit_status()
    print(f"rc={rc}\n", flush=True)
    return rc


def have_numpy():
    rc = run("python3 -c 'import numpy' 2>/dev/null && echo HAVE_NUMPY")
    return rc == 0


# Step 1: try sudo apt install with password via -S
print("=== sudo apt install python3-{pip,numpy,pandas,scipy,sklearn} (password via -S) ===")
run(
    f"set -o pipefail; "
    f"echo '{PASS}' | sudo -S apt-get install -y python3-pip python3-numpy python3-pandas python3-scipy python3-sklearn 2>&1 | tail -20",
    t=900,
)

if have_numpy():
    print("OK numpy available after apt")
else:
    # Step 2: bootstrap pip via get-pip.py and install user-site packages
    print("\n=== fallback: bootstrap pip via get-pip.py ===")
    run("curl -sSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && python3 /tmp/get-pip.py --user 2>&1 | tail -8", t=300)
    run("python3 -m pip install --user --quiet numpy pandas scipy scikit-learn 2>&1 | tail -8", t=600)

print("\n=== verify versions ===")
run('python3 -c "import numpy, pandas, scipy, sklearn; '
    'print(\\"numpy\\", numpy.__version__, \\"pandas\\", pandas.__version__, '
    '\\"scipy\\", scipy.__version__, \\"sklearn\\", sklearn.__version__)"')

print("=== run baseline eval ===")
run("cd /home/fteam6/project/rogii-wellbore-geology-prediction && python3 -u scripts/eval_baselines_remote.py 2>&1 | tee logs/baseline_eval.log",
    t=2400)

cli.close()
