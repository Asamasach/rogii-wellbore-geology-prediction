"""Upload fetcher to fteam6 and launch fully-detached background download."""
import os
import sys

import paramiko

HOST = os.environ["FTEAM6_HOST"]
PORT = int(os.environ["FTEAM6_PORT"])
USER = os.environ["FTEAM6_USER"]
PASS = os.environ["FTEAM6_PASS"]
TOKEN = os.environ["KAGGLE_API_TOKEN"]
REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction"


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    return cli


def run(cmd, cli, timeout=60, stream=True):
    print(f"$ {cmd}", flush=True)
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    if stream:
        for line in iter(stdout.readline, ""):
            if not line:
                break
            try:
                print(line.rstrip(), flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode().rstrip(), flush=True)
    else:
        out = stdout.read().decode(errors="replace")
        if out:
            try:
                print(out)
            except UnicodeEncodeError:
                print(out.encode("ascii", "replace").decode())
    err = stderr.read().decode(errors="replace").rstrip()
    if err:
        print(f"[stderr] {err}", flush=True)
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
        sftp.put(local, remote)
        print(f"uploaded {local} -> {remote} ({os.path.getsize(local)} bytes)", flush=True)
    finally:
        sftp.close()


cli = connect()
try:
    print("=== upload fetcher ===")
    put(
        r"c:\projects\kaggle\birdclef-2026\scripts\fetch_kaggle_data.py",
        f"{REMOTE}/scripts/fetch_kaggle_data.py",
        cli,
    )

    print("=== launch background download (workers=4, gentle) ===")
    launch = (
        f"cd {REMOTE} && "
        f"export KAGGLE_API_TOKEN={TOKEN} && "
        f"( setsid nohup python3 -u scripts/fetch_kaggle_data.py "
        f"--comp rogii-wellbore-geology-prediction --dest data/raw "
        f"--workers 4 "
        f"< /dev/null > logs/fetch.log 2>&1 & "
        f"echo $! > logs/fetch.pid ) && "
        f"sleep 1 && cat logs/fetch.pid"
    )
    run(launch, cli, stream=False, timeout=20)

    print("=== confirm process is alive ===")
    run(
        f"sleep 4 && "
        f"PID=$(cat {REMOTE}/logs/fetch.pid); "
        f"ps -o pid,etime,cmd -p $PID 2>&1 | head -3; "
        f"echo ---log---; "
        f"head -15 {REMOTE}/logs/fetch.log",
        cli,
        stream=False,
        timeout=20,
    )
finally:
    cli.close()
