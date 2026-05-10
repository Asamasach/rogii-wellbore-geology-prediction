"""Ship + install catboost + run on fteam11."""
import os, paramiko
from pathlib import Path

J_HOST=os.environ["FTEAM6_HOST"]; J_PORT=int(os.environ["FTEAM6_PORT"])
J_USER=os.environ["FTEAM6_USER"]; J_PASS=os.environ["FTEAM6_PASS"]
T_HOST=os.environ["FTEAM11_HOST"]; T_PORT=int(os.environ["FTEAM11_PORT"])
T_USER=os.environ["FTEAM11_USER"]; T_PASS=os.environ["FTEAM11_PASS"]

j=paramiko.SSHClient(); j.set_missing_host_key_policy(paramiko.AutoAddPolicy())
j.connect(J_HOST, port=J_PORT, username=J_USER, password=J_PASS, timeout=15)
chan=j.get_transport().open_channel("direct-tcpip",(T_HOST,T_PORT),("127.0.0.1",0))
t11=paramiko.SSHClient(); t11.set_missing_host_key_policy(paramiko.AutoAddPolicy())
t11.connect(T_HOST, port=T_PORT, username=T_USER, password=T_PASS, timeout=15, sock=chan)

REMOTE="/home/fteam11/projects/rogii-wellbore-geology-prediction"
LOCAL=Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\scripts")
sftp=t11.open_sftp()
for fname in ("eval_catboost_remote.py",):
    local=LOCAL/fname
    sftp.put(str(local), f"{REMOTE}/scripts/{fname}")
    print(f"uploaded {fname} ({local.stat().st_size} bytes)")
sftp.close()


def run(c, t=3600):
    print(f"$ {c[:200]}", flush=True)
    _, o, e = t11.exec_command(c, timeout=t)
    for ln in iter(o.readline, ""):
        if not ln: break
        try: print(ln.rstrip(), flush=True)
        except UnicodeEncodeError: print(ln.encode("ascii","replace").decode().rstrip(), flush=True)
    err=e.read().decode(errors="replace").rstrip()
    if err:
        try: print(f"[stderr] {err}", flush=True)
        except UnicodeEncodeError: print("[stderr]", err.encode("ascii","replace").decode())
    rc=o.channel.recv_exit_status()
    print(f"rc={rc}\n", flush=True)


print("\n=== ensure catboost installed ===")
run("python3 -c 'import catboost; print(\"catboost\", catboost.__version__)' 2>/dev/null || pip3 install --user --quiet catboost 2>&1 | tail -5", t=600)
print("\n=== run catboost training + stack ===")
run(f"cd {REMOTE} && python3 -u scripts/eval_catboost_remote.py 2>&1 | tee logs/catboost_v3.log", t=3600)

t11.close(); j.close()
