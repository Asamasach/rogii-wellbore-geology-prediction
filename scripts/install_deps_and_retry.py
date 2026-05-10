"""Install numpy/pandas on fteam6 then re-run baseline eval."""
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


def run(c, t=600, stream=True):
    print(f"$ {c[:200]}{'...' if len(c) > 200 else ''}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t)
    if stream:
        for ln in iter(o.readline, ""):
            if not ln:
                break
            try: print(ln.rstrip(), flush=True)
            except UnicodeEncodeError: print(ln.encode("ascii", "replace").decode().rstrip(), flush=True)
    else:
        out = o.read().decode(errors="replace")
        if out: print(out)
    err = e.read().decode(errors="replace").rstrip()
    if err:
        try: print(f"[stderr] {err}", flush=True)
        except UnicodeEncodeError: print("[stderr]", err.encode("ascii", "replace").decode())
    rc = o.channel.recv_exit_status()
    print(f"rc={rc}\n", flush=True)
    return rc


print("=== check what python tools are available ===")
run("which python3 python pip3 pip; python3 -m ensurepip --version 2>/dev/null; ls /usr/bin/pip* 2>/dev/null", t=30)

print("=== try install numpy + pandas via python3 -m pip --user ===")
run("python3 -m pip install --user --quiet numpy pandas 2>&1 | tail -10; "
    "python3 -c 'import numpy, pandas; print(\"numpy\", numpy.__version__, \"pandas\", pandas.__version__)'",
    t=600)

print("=== re-run baseline eval ===")
run("cd /home/fteam6/project/rogii-wellbore-geology-prediction && "
    "python3 -u scripts/eval_baselines_remote.py 2>&1 | tee logs/baseline_eval.log",
    t=2400)

cli.close()
