"""Set up rogii project on fteam6, install kaggle deps, persist token, smoke-test auth.

Self-contained — inlines paramiko, reads creds from env vars FTEAM6_HOST/PORT/USER/PASS
and KAGGLE_API_TOKEN. No hardcoded credentials.
"""
import os
import sys
from pathlib import Path

import paramiko

HOST = os.environ.get("FTEAM6_HOST")
PORT = int(os.environ.get("FTEAM6_PORT", "0") or 0)
USER = os.environ.get("FTEAM6_USER")
PASS = os.environ.get("FTEAM6_PASS")
TOKEN = os.environ.get("KAGGLE_API_TOKEN")
if not all([HOST, PORT, USER, PASS, TOKEN]):
    sys.exit("error: set FTEAM6_HOST / FTEAM6_PORT / FTEAM6_USER / FTEAM6_PASS / KAGGLE_API_TOKEN")

REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction"


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    return cli


def run(cmd, cli, timeout=600):
    print(f"$ {cmd}", flush=True)
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    for line in iter(stdout.readline, ""):
        if not line:
            break
        try:
            print(line.rstrip(), flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode().rstrip(), flush=True)
    err = stderr.read().decode(errors="replace").rstrip()
    if err:
        try:
            print(f"[stderr] {err}", flush=True)
        except UnicodeEncodeError:
            print("[stderr]", err.encode("ascii", "replace").decode(), flush=True)
    rc = stdout.channel.recv_exit_status()
    print(f"rc={rc}\n", flush=True)
    return rc


def put(local, remote, cli):
    sftp = cli.open_sftp()
    try:
        d = os.path.dirname(remote)
        if d:
            try:
                sftp.stat(d)
            except IOError:
                p = ""
                for part in d.strip("/").split("/"):
                    p = p + "/" + part
                    try:
                        sftp.stat(p)
                    except IOError:
                        sftp.mkdir(p)
        sftp.put(str(local), remote)
        print(f"uploaded {local} -> {remote} ({os.path.getsize(local)} bytes)", flush=True)
    finally:
        sftp.close()


def main():
    cli = connect()
    try:
        print("=== make project dirs ===")
        run(
            f"mkdir -p {REMOTE}/data/raw {REMOTE}/data/features {REMOTE}/scripts "
            f"{REMOTE}/models/oof {REMOTE}/models/pte {REMOTE}/submissions/archived "
            f"{REMOTE}/logs && ls -la {REMOTE}/",
            cli,
        )
        print("=== install kaggle CLI (Python 3.10 -> kaggle 2.x supported) ===")
        run("pip3 install --user --quiet --upgrade kaggle 2>&1 | tail -3 && /home/fteam6/.local/bin/kaggle --version", cli, timeout=300)
        print("=== persist KAGGLE_API_TOKEN on fteam6 (~/.config/kaggle_env, mode 600) ===")
        setup = (
            f"mkdir -p ~/.config && "
            f"echo 'export KAGGLE_API_TOKEN={TOKEN}' > ~/.config/kaggle_env && "
            f"chmod 600 ~/.config/kaggle_env && "
            f"if ! grep -q kaggle_env ~/.bashrc 2>/dev/null; then "
            f"  echo '[ -f ~/.config/kaggle_env ] && . ~/.config/kaggle_env' >> ~/.bashrc; "
            f"fi && ls -la ~/.config/kaggle_env"
        )
        run(setup, cli)
        print("=== auth + rate-limit smoke test (small file) ===")
        run(
            f'export KAGGLE_API_TOKEN={TOKEN} && '
            f'curl -sS -L -D - -o /tmp/_ss.csv '
            f'-H "Authorization: Bearer {TOKEN}" '
            f'"https://www.kaggle.com/api/v1/competitions/data/download/'
            f'rogii-wellbore-geology-prediction/sample_submission.csv" '
            f'| head -3; '
            f'ls -la /tmp/_ss.csv; head -2 /tmp/_ss.csv',
            cli,
        )
    finally:
        cli.close()


if __name__ == "__main__":
    main()
