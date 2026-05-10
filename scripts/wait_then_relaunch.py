"""Wait until Kaggle rate limit clears, then relaunch the download with strict throttle.

Polls the smoke endpoint every 60s. Once it returns non-429, uploads the latest
fetcher and launches a workers=1 + delay=1.0 download. Then polls the running
download every 90s until [done] appears in the log.
"""
import os
import sys
import time

import paramiko

HOST = os.environ["FTEAM6_HOST"]; PORT = int(os.environ["FTEAM6_PORT"])
USER = os.environ["FTEAM6_USER"]; PASS = os.environ["FTEAM6_PASS"]
TOKEN = os.environ["KAGGLE_API_TOKEN"]
REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction"


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    return cli


def cmd(cli, c, t=60):
    stdin, stdout, stderr = cli.exec_command(c, timeout=t)
    return stdout.read().decode(errors="replace"), stdout.channel.recv_exit_status()


def put(cli, local, remote):
    sftp = cli.open_sftp()
    try:
        sftp.put(local, remote)
    finally:
        sftp.close()


def probe_rate(cli):
    out, _ = cmd(
        cli,
        f'curl -sS -o /dev/null -w "%{{http_code}}" '
        f'-H "Authorization: Bearer {TOKEN}" '
        f'"https://www.kaggle.com/api/v1/competitions/data/download/'
        f'rogii-wellbore-geology-prediction/sample_submission.csv"',
        t=30,
    )
    return out.strip()


def main():
    cli = connect()
    print("[wait] polling rate-limit state every 60s", flush=True)
    poll_count = 0
    while True:
        try:
            code = probe_rate(cli)
        except Exception as e:
            print(f"[probe-err] {type(e).__name__}: {e}", flush=True)
            try: cli.close()
            except: pass
            time.sleep(60)
            cli = connect()
            continue
        poll_count += 1
        print(f"[wait] poll #{poll_count}: code={code}", flush=True)
        if code in ("200", "302"):
            print("[wait] rate-limit cleared", flush=True)
            break
        time.sleep(60)

    # Upload latest fetcher
    print("[launch] uploading fetcher", flush=True)
    put(cli, r"c:\projects\kaggle\birdclef-2026\scripts\fetch_kaggle_data.py",
        f"{REMOTE}/scripts/fetch_kaggle_data.py")

    # Clear stale partial state
    cmd(cli, f"rm -f {REMOTE}/data/raw/_failures.log; "
             f"find {REMOTE}/data/raw -name '*.part' -delete 2>/dev/null; echo cleaned")

    # Launch fully detached, workers=1 delay=1.0 (= 1 req/s with download time)
    launch = (
        f"cd {REMOTE} && "
        f"export KAGGLE_API_TOKEN={TOKEN} && "
        f"( setsid nohup python3 -u scripts/fetch_kaggle_data.py "
        f"--comp rogii-wellbore-geology-prediction --dest data/raw "
        f"--workers 1 --delay 1.0 "
        f"< /dev/null > logs/fetch.log 2>&1 & "
        f"echo $! > logs/fetch.pid ) && "
        f"sleep 1 && cat logs/fetch.pid"
    )
    out, rc = cmd(cli, launch, t=20)
    print(f"[launch] pid={out.strip()} rc={rc}", flush=True)

    # Tail completion
    print("[watch] polling every 90s until [done]", flush=True)
    last_count = -1; stall = 0
    t0 = time.time()
    while True:
        if time.time() - t0 > 4 * 3600:
            print("[watch-timeout] giving up after 4h", flush=True)
            break
        try:
            out, _ = cmd(
                cli,
                f"PID=$(cat {REMOTE}/logs/fetch.pid 2>/dev/null); "
                f"if [ -n \"$PID\" ] && ps -p $PID > /dev/null 2>&1; then alive=1; else alive=0; fi; "
                f"line=$(tail -1 {REMOTE}/logs/fetch.log 2>/dev/null); "
                f"count=$(find {REMOTE}/data/raw -type f ! -name '_manifest.json' | wc -l); "
                f"size=$(du -sm {REMOTE}/data/raw | awk '{{print $1}}'); "
                f"echo alive=$alive count=$count size_mb=$size; echo last=$line",
                t=30,
            )
        except Exception as e:
            print(f"[probe-err] {type(e).__name__}: {e}", flush=True)
            try: cli.close()
            except: pass
            time.sleep(90)
            cli = connect()
            continue

        alive="0"; count="?"; size="?"; last=""
        for ln in out.splitlines():
            if ln.startswith("alive="):
                p = dict(x.split("=",1) for x in ln.split())
                alive=p.get("alive","0"); count=p.get("count","?"); size=p.get("size_mb","?")
            elif ln.startswith("last="):
                last = ln[5:]
        try: print(f"[watch] alive={alive} files={count} size_mb={size}  log: {last}", flush=True)
        except UnicodeEncodeError: print(f"[watch] alive={alive} files={count}", flush=True)
        if alive == "0":
            print("[finished]" if "[done]" in last else "[crashed-or-rerun-needed]", flush=True)
            break
        try:
            cnt = int(count)
            if cnt == last_count:
                stall += 1
                if stall >= 8:  # 12 min of no progress
                    print("[stalled] 8 polls = 12 min no new files", flush=True); break
            else:
                stall = 0; last_count = cnt
        except ValueError:
            pass
        time.sleep(90)

    cli.close()


if __name__ == "__main__":
    main()
