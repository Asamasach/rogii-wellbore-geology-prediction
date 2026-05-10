"""EDA on the 514 files we already pulled to fteam6 — sample_submission + a few wells."""
import os
import sys
import paramiko

HOST = os.environ["FTEAM6_HOST"]; PORT = int(os.environ["FTEAM6_PORT"])
USER = os.environ["FTEAM6_USER"]; PASS = os.environ["FTEAM6_PASS"]
REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)

def run(cmd, t=60):
    print(f"$ {cmd}", flush=True)
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=t)
    out = stdout.read().decode(errors="replace")
    if out:
        try: print(out)
        except UnicodeEncodeError: print(out.encode("ascii", "replace").decode())
    err = stderr.read().decode(errors="replace")
    if err: print(f"[stderr] {err}")
    rc = stdout.channel.recv_exit_status()
    print(f"rc={rc}\n")

try:
    print("=== sample_submission.csv (head + tail + structure) ===")
    run(f"head -5 {REMOTE}/sample_submission.csv && echo ... && tail -3 {REMOTE}/sample_submission.csv && echo ---; wc -l {REMOTE}/sample_submission.csv")

    print("=== files we have so far (broken down by type/split) ===")
    run(
        f"find {REMOTE} -type f ! -name '_manifest.json' | "
        f"awk -F/ '{{ "
        f"  if (index($0, \"/test/\")) split=\"test\"; "
        f"  else if (index($0, \"/train/\")) split=\"train\"; "
        f"  else split=\"root\"; "
        f"  if (match($NF, /__([^.]+)\\./, a)) kind=a[1]; else kind=\"-\"; "
        f"  print split\"/\"kind "
        f"}}' | sort | uniq -c | sort -rn"
    )

    print("=== sample horizontal_well.csv (head and column count) ===")
    run(
        f"f=$(find {REMOTE}/train -name '*horizontal_well.csv' 2>/dev/null | head -1); "
        f"echo file=$f; head -3 \"$f\" 2>/dev/null; "
        f"echo cols=$(head -1 \"$f\" 2>/dev/null | awk -F, '{{print NF}}'); "
        f"echo rows=$(wc -l < \"$f\" 2>/dev/null)"
    )

    print("=== sample typewell.csv (head and column count) ===")
    run(
        f"f=$(find {REMOTE}/train -name '*typewell.csv' 2>/dev/null | head -1); "
        f"echo file=$f; head -3 \"$f\" 2>/dev/null; "
        f"echo cols=$(head -1 \"$f\" 2>/dev/null | awk -F, '{{print NF}}'); "
        f"echo rows=$(wc -l < \"$f\" 2>/dev/null)"
    )

    print("=== test/ dir (only 6 files total) ===")
    run(f"find {REMOTE}/test -type f -exec ls -la {{}} \\;")

    print("=== how many unique well_ids do we have train files for? ===")
    run(
        f"find {REMOTE}/train -type f 2>/dev/null | "
        f"awk -F/ '{{print $NF}}' | awk -F__ '{{print $1}}' | sort -u | wc -l"
    )
finally:
    cli.close()
