"""Upload predict_tvt_lgb.py + eval_lgb_remote.py to fteam11 and run via fteam6 jump."""
import os
import sys
import paramiko
from pathlib import Path

J_HOST = os.environ["FTEAM6_HOST"]; J_PORT = int(os.environ["FTEAM6_PORT"])
J_USER = os.environ["FTEAM6_USER"]; J_PASS = os.environ["FTEAM6_PASS"]
T_HOST = os.environ["FTEAM11_HOST"]; T_PORT = int(os.environ["FTEAM11_PORT"])
T_USER = os.environ["FTEAM11_USER"]; T_PASS = os.environ["FTEAM11_PASS"]

LOCAL_SCRIPTS = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\scripts")
REMOTE = "/home/fteam11/projects/rogii-wellbore-geology-prediction"


def jump_connect():
    j = paramiko.SSHClient()
    j.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    j.connect(J_HOST, port=J_PORT, username=J_USER, password=J_PASS, timeout=15)
    chan = j.get_transport().open_channel("direct-tcpip", (T_HOST, T_PORT), ("127.0.0.1", 0))
    t = paramiko.SSHClient()
    t.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t.connect(T_HOST, port=T_PORT, username=T_USER, password=T_PASS, timeout=15, sock=chan)
    return j, t


def run(cli, c, t=2400):
    print(f"$ {c[:200]}{'...' if len(c) > 200 else ''}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t)
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


j, t11 = jump_connect()

# upload via SFTP through the jump channel
sftp = t11.open_sftp()
try:
    for fname in ("predict_tvt_lgb.py", "eval_lgb_remote.py"):
        local = LOCAL_SCRIPTS / fname
        remote = f"{REMOTE}/scripts/{fname}"
        sftp.put(str(local), remote)
        print(f"uploaded {fname} ({local.stat().st_size} bytes)")
finally:
    sftp.close()

print("\n=== run eval ===")
run(t11,
    f"cd {REMOTE} && python3 -u scripts/eval_lgb_remote.py 2>&1 | tee logs/lgb_v1_eval.log",
    t=2400)

t11.close(); j.close()
