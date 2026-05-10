"""Poll fteam6 download until it completes; emit one event per poll."""
import os
import sys
import time

import paramiko

HOST = os.environ["FTEAM6_HOST"]
PORT = int(os.environ["FTEAM6_PORT"])
USER = os.environ["FTEAM6_USER"]
PASS = os.environ["FTEAM6_PASS"]
REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction"
POLL = 60
TIMEOUT = 60 * 60  # 1 hour cap


def connect():
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)
    return cli


def probe(cli):
    cmd = (
        f"PID=$(cat {REMOTE}/logs/fetch.pid 2>/dev/null); "
        f"if [ -n \"$PID\" ] && ps -p $PID > /dev/null 2>&1; then alive=1; else alive=0; fi; "
        f"line=$(tail -1 {REMOTE}/logs/fetch.log 2>/dev/null); "
        f"count=$(find {REMOTE}/data/raw -type f ! -name '_manifest.json' 2>/dev/null | wc -l); "
        f"size=$(du -sm {REMOTE}/data/raw 2>/dev/null | awk '{{print $1}}'); "
        f"echo \"alive=$alive count=$count size_mb=$size\"; "
        f"echo \"last=$line\""
    )
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=30)
    return stdout.read().decode(errors="replace")


cli = connect()
t0 = time.time()
last_count = -1
stall_strikes = 0
try:
    while True:
        if time.time() - t0 > TIMEOUT:
            print("[timeout] watcher exceeded 1h, exiting", flush=True)
            break
        try:
            out = probe(cli)
        except Exception as e:
            print(f"[probe-error] {type(e).__name__}: {e}", flush=True)
            time.sleep(POLL)
            try:
                cli.close()
            except Exception:
                pass
            cli = connect()
            continue

        alive = "0"; count = "?"; size_mb = "?"; last = ""
        for line in out.splitlines():
            if line.startswith("alive="):
                parts = dict(p.split("=", 1) for p in line.split())
                alive = parts.get("alive", "0")
                count = parts.get("count", "?")
                size_mb = parts.get("size_mb", "?")
            elif line.startswith("last="):
                last = line[5:]
        try:
            print(f"[poll] alive={alive} files={count} size_mb={size_mb}  log: {last}", flush=True)
        except UnicodeEncodeError:
            print(f"[poll] alive={alive} files={count} size_mb={size_mb}", flush=True)

        if alive == "0":
            print("[finished]" if "[done]" in last else "[crashed]", flush=True)
            break

        try:
            cnt = int(count)
            if cnt == last_count:
                stall_strikes += 1
                if stall_strikes >= 5:
                    print("[stalled] 5 polls with no new files", flush=True)
                    break
            else:
                stall_strikes = 0
                last_count = cnt
        except ValueError:
            pass
        time.sleep(POLL)
finally:
    try:
        cli.close()
    except Exception:
        pass
