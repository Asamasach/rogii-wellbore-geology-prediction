"""Quick status check: fteam6 download state + current Kaggle rate-limit state."""
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

R = "/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw"
TOKEN = os.environ["KAGGLE_API_TOKEN"]


def run(c):
    _, o, _ = cli.exec_command(c, timeout=30)
    return o.read().decode(errors="replace").strip()


print("=== fteam6 state ===")
print("total files:       ", run(f"find {R} -type f ! -name _manifest.json | wc -l"))
print("total size MB:     ", run(f"du -sm {R} | cut -f1"))
print("train typewells:   ", run(f"find {R}/train -name '*typewell.csv' 2>/dev/null | wc -l"), "/ 1546")
print("train horizontals: ", run(f"find {R}/train -name '*horizontal_well.csv' 2>/dev/null | wc -l"), "/ ~773")
print("test files:        ", run(f"find {R}/test -type f 2>/dev/null | wc -l"), "/ 6")
print(".pptx:             ", run(f"ls {R}/AI_*.pptx 2>/dev/null | wc -l"), "/ 1")
print("sample_submission: ", run(f"ls {R}/sample_submission.csv 2>/dev/null | wc -l"), "/ 1")
print()
print("=== any background python on fteam6 still running? ===")
print(run("ps -eo pid,etime,cmd | grep -E 'fetch_kaggle|wait_then|python3 -u scripts' | grep -v grep || echo 'no python jobs'"))
print()
print("=== Kaggle rate-limit probe (single GET) ===")
print(run(
    f'curl -sS -o /tmp/_p.bin '
    f'-w "http_code=%{{http_code}} size=%{{size_download}}\\n" '
    f'-D /tmp/_h.txt '
    f'-H "Authorization: Bearer {TOKEN}" '
    f'"https://www.kaggle.com/api/v1/competitions/data/download/'
    f'rogii-wellbore-geology-prediction/sample_submission.csv"'
))
print("--- response headers (status + retry-after) ---")
print(run("head -1 /tmp/_h.txt; grep -i '^retry-after' /tmp/_h.txt 2>/dev/null || echo 'no retry-after header'"))

cli.close()
