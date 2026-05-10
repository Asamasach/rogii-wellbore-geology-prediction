"""Clean /tmp probe leftovers on fteam6."""
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


def run(c):
    print(f"$ {c}", flush=True)
    _, o, e = cli.exec_command(c, timeout=60)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    rc = o.channel.recv_exit_status()
    if out: print(out.encode("ascii", "replace").decode())
    if err: print(f"[stderr] {err.encode('ascii', 'replace').decode()}")
    print(f"rc={rc}\n")


# Be specific to avoid touching anything else in /tmp
run("rm -f /tmp/_h.txt /tmp/_nh.txt /tmp/_oh.txt /tmp/_nt.bin /tmp/_p.bin /tmp/_ss.csv /tmp/_t.csv /tmp/_probe.py /tmp/get-pip.py && echo deleted")
run("ls /tmp/_*.csv /tmp/_*.txt /tmp/_*.bin /tmp/_*.py /tmp/get-pip.py 2>/dev/null; echo '---if nothing above, /tmp is clean---'")

cli.close()
