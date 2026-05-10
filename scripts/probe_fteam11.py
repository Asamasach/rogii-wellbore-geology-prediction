"""Probe a remote host's reachability + specs (creds from env vars)."""
import os
import socket
import sys
import paramiko

HOST = os.environ.get("FTEAM11_HOST")
USER = os.environ.get("FTEAM11_USER")
PASS = os.environ.get("FTEAM11_PASS")
if not (HOST and USER and PASS):
    sys.exit("error: set FTEAM11_HOST / FTEAM11_USER / FTEAM11_PASS env vars")

# Try common SSH ports
for port in (22, 50001, 50003, 50011, 22011):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect((HOST, port))
        print(f"TCP open: {HOST}:{port}")
    except Exception as e:
        print(f"TCP closed/timeout: {HOST}:{port}  ({type(e).__name__})")
    finally:
        s.close()

# Try SSH on the open ports
for port in (22, 50001, 50003, 50011, 22011):
    print(f"\n=== SSH attempt {USER}@{HOST}:{port} ===")
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        cli.connect(HOST, port=port, username=USER, password=PASS, timeout=10)
        print("  SSH OK")
    except Exception as e:
        print(f"  SSH FAILED: {type(e).__name__}: {str(e)[:160]}")
        continue

    def run(c, t=30):
        _, o, e = cli.exec_command(c, timeout=t)
        out = o.read().decode(errors="replace").rstrip()
        err = e.read().decode(errors="replace").rstrip()
        rc = o.channel.recv_exit_status()
        safe = lambda s: s.encode("ascii", "replace").decode()
        print(f"$ {c}")
        if out: print(safe(out))
        if err: print(f"[stderr] {safe(err)}")
        print(f"rc={rc}")

    run("whoami && hostname && uname -srm")
    run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo 'no nvidia-smi'")
    run("df -BG / | tail -1; free -g | head -2")
    run("python3 --version; which python3 pip3 2>/dev/null; ls /usr/bin/pip* 2>/dev/null")
    run("ls -la ~/")
    cli.close()
    print(f"=== {port} DONE ===")
    sys.exit(0)

print("\nNo working SSH port found")
