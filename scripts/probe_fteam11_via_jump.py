"""Probe fteam11 reachability + specs via jump host fteam6."""
import os
import socket
import sys
import paramiko

# Jump = fteam6
J_HOST = os.environ["FTEAM6_HOST"]
J_PORT = int(os.environ["FTEAM6_PORT"])
J_USER = os.environ["FTEAM6_USER"]
J_PASS = os.environ["FTEAM6_PASS"]
# Target = fteam11
T_HOST = os.environ["FTEAM11_HOST"]
T_PASS = os.environ["FTEAM11_PASS"]
T_USER = os.environ["FTEAM11_USER"]


def jump_connect(t_port=22):
    j = paramiko.SSHClient()
    j.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    j.connect(J_HOST, port=J_PORT, username=J_USER, password=J_PASS, timeout=15)
    chan = j.get_transport().open_channel("direct-tcpip", (T_HOST, t_port), ("127.0.0.1", 0))
    t = paramiko.SSHClient()
    t.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t.connect(T_HOST, port=t_port, username=T_USER, password=T_PASS, timeout=15, sock=chan)
    return j, t


def run(cli, c, t=30):
    print(f"$ {c}")
    _, o, e = cli.exec_command(c, timeout=t)
    out = o.read().decode(errors="replace").rstrip()
    err = e.read().decode(errors="replace").rstrip()
    rc = o.channel.recv_exit_status()
    safe = lambda s: s.encode("ascii", "replace").decode()
    if out: print(safe(out))
    if err: print(f"[stderr] {safe(err)}")
    print(f"rc={rc}\n")


# Probe fteam11 via different ports
for port in (22, 2222, 50011, 50001):
    print(f"=== try fteam11 SSH on :{port} via fteam6 jump ===")
    try:
        j, t = jump_connect(t_port=port)
        print(f"  SSH OK on :{port}")
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {str(e)[:160]}\n")
        continue

    # Got it — probe specs
    run(t, "whoami && hostname && uname -srm")
    run(t, "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo 'no nvidia-smi'")
    run(t, "df -BG / 2>/dev/null | tail -1; free -g | head -2")
    run(t, "python3 --version; which python3 pip3 pip 2>/dev/null")
    run(t, "ls -la ~/")

    # Save the working port for future use
    print(f"\nFTEAM11_PORT_WORKING = {port}")
    t.close(); j.close()
    sys.exit(0)
print("\nall ports failed")
