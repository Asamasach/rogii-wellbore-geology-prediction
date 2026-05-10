"""Thorough sweep of fteam6 for any artifacts left from this work."""
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


print("=== /tmp probe leftovers ===")
run("ls -la /tmp/_*.csv /tmp/_*.txt /tmp/_*.bin /tmp/_*.py /tmp/get-pip.py /tmp/_h.txt /tmp/_oh.txt /tmp/_nt.bin /tmp/_nh.txt /tmp/_p.bin /tmp/_t.csv /tmp/_probe.py /tmp/_ss.csv 2>/dev/null; echo ---")

print("=== ~/.local pip-user packages (kaggle, kagglehub were installed there) ===")
run("ls ~/.local/bin 2>/dev/null | head -30; echo ---; "
    "[ -d ~/.local/lib ] && du -sh ~/.local/lib 2>/dev/null; "
    "[ -d ~/.local/lib ] && ls ~/.local/lib/python3.10/site-packages 2>/dev/null | grep -iE 'kaggle|kagglehub' | head -5")

print("=== bash_history — does it leak the KGAT token? ===")
run("grep -cE 'KGAT_[a-zA-Z0-9]+' ~/.bash_history 2>/dev/null || echo 0_matches; "
    "echo ---first-3-matching-lines---; "
    "grep -nE 'KGAT_[a-zA-Z0-9]+' ~/.bash_history 2>/dev/null | head -3 | sed 's/KGAT_[a-zA-Z0-9]*/KGAT_<REDACTED>/g'")

print("=== other config / cache dirs we may have touched ===")
run("ls -la ~/.kaggle 2>/dev/null; ls -la ~/.config/kaggle 2>/dev/null; "
    "ls -la ~/.cache 2>/dev/null | head -10")

target_host = os.environ.get("FTEAM11_HOST", "")
print(f"=== known_hosts entries we added ({target_host or '<unset>'}) ===")
run(f"grep -c '{target_host}' ~/.ssh/known_hosts 2>/dev/null || echo no_known_hosts; "
    f"grep -n '{target_host}' ~/.ssh/known_hosts 2>/dev/null | head -3 | cut -c1-40")

cli.close()
