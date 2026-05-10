"""Inventory fteam6 before any deletion."""
import os
import paramiko

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(
    os.environ["FTEAM6_HOST"],
    port=int(os.environ["FTEAM6_PORT"]),
    username=os.environ["FTEAM6_USER"],
    password=os.environ["FTEAM6_PASS"],
    timeout=15,
)


def run(c, t=120):
    print(f"$ {c}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    rc = o.channel.recv_exit_status()
    safe = lambda s: s.encode("ascii", "replace").decode()
    if out: print(safe(out))
    if err: print(f"[stderr] {safe(err)}")
    print(f"rc={rc}\n")


print("=== ~/project (top level) ===")
run("ls -la ~/project/")
print("=== sizes per project subdir ===")
run("du -sh ~/project/*/ 2>/dev/null")
print("=== geohab / birdclef references? ===")
run("ls ~/project/ | grep -iE 'geohab|birdclef|test|rogii'")
print("=== any kaggle creds anywhere ===")
run("ls -la ~/.config/kaggle_env 2>/dev/null; ls -la ~/.kaggle 2>/dev/null; ls -la ~/.config/kaggle 2>/dev/null")
print("=== relevant .bashrc lines ===")
run("grep -nE 'kaggle|KAGGLE' ~/.bashrc 2>/dev/null || echo no_kaggle_in_bashrc")
print("=== overall disk ===")
run("df -BG /home /")

cli.close()
