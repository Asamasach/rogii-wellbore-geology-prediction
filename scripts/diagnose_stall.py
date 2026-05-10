"""Kill current download, inspect failures, probe single request to test current rate-limit state."""
import os
import sys

import paramiko

HOST = os.environ["FTEAM6_HOST"]; PORT = int(os.environ["FTEAM6_PORT"])
USER = os.environ["FTEAM6_USER"]; PASS = os.environ["FTEAM6_PASS"]
TOKEN = os.environ["KAGGLE_API_TOKEN"]
REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)

def run(cmd, t=60):
    print(f"$ {cmd}", flush=True)
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=t)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    if out:
        try: print(out)
        except UnicodeEncodeError: print(out.encode("ascii", "replace").decode())
    if err: print(f"[stderr] {err}")
    print(f"rc={rc}\n")

try:
    print("=== kill download ===")
    run(f"PID=$(cat {REMOTE}/logs/fetch.pid 2>/dev/null); "
        f"if [ -n \"$PID\" ] && ps -p $PID > /dev/null; then kill -9 $PID; echo killed_$PID; else echo not_running; fi")

    print("=== final progress line ===")
    run(f"tail -3 {REMOTE}/logs/fetch.log")

    print("=== current single-request probe (rate-limit state NOW) ===")
    run(f'curl -sS -L -o /dev/null '
        f'-w "code=%{{http_code}} size=%{{size_download}}\\n" '
        f'-D /tmp/_h.txt '
        f'-H "Authorization: Bearer {TOKEN}" '
        f'"https://www.kaggle.com/api/v1/competitions/data/download/'
        f'rogii-wellbore-geology-prediction/sample_submission.csv"; '
        f'echo --- response headers ---; cat /tmp/_h.txt | head -10')

    print("=== current files on disk ===")
    run(f"echo files: $(find {REMOTE}/data/raw -type f ! -name '_manifest.json' | wc -l); "
        f"echo size:  $(du -sm {REMOTE}/data/raw | awk '{{print $1}}') MB")
finally:
    cli.close()
