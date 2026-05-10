"""Upload predict_tvt.py + eval_baselines_remote.py to fteam6 and run the eval."""
import os
import paramiko
from pathlib import Path

LOCAL_SCRIPTS = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\scripts")
REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction/scripts"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(
    os.environ["FTEAM6_HOST"],
    port=int(os.environ["FTEAM6_PORT"]),
    username=os.environ["FTEAM6_USER"],
    password=os.environ["FTEAM6_PASS"],
    timeout=15,
)


def run(c, t=600):
    _, o, e = cli.exec_command(c, timeout=t)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    rc = o.channel.recv_exit_status()
    safe = lambda s: s.encode("ascii", "replace").decode()
    if out: print(safe(out))
    if err: print(f"[stderr] {safe(err)}")
    print(f"rc={rc}\n")
    return rc, out


# Upload scripts
sftp = cli.open_sftp()
try:
    for fname in ("predict_tvt.py", "eval_baselines_remote.py"):
        local = LOCAL_SCRIPTS / fname
        remote_path = f"{REMOTE}/{fname}"
        sftp.put(str(local), remote_path)
        print(f"uploaded {local} -> {remote_path} ({local.stat().st_size} bytes)")
finally:
    sftp.close()

print("\n=== run eval on fteam6 ===")
run(
    f"cd /home/fteam6/project/rogii-wellbore-geology-prediction && "
    f"python3 -u scripts/eval_baselines_remote.py 2>&1 | tee logs/baseline_eval.log",
    t=2400,
)

cli.close()
