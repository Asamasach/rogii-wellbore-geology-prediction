"""Update fteam6's persisted token and probe rate-limit state with new token."""
import os
import paramiko

NEW = os.environ["KAGGLE_API_TOKEN"]
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
    _, o, e = cli.exec_command(c, timeout=30)
    return o.read().decode(errors="replace").strip(), e.read().decode(errors="replace").strip()


print("--- update ~/.config/kaggle_env on fteam6 ---")
out, _ = run(
    f'echo "export KAGGLE_API_TOKEN={NEW}" > ~/.config/kaggle_env && '
    f'chmod 600 ~/.config/kaggle_env && echo updated && cat ~/.config/kaggle_env'
)
print(out)

print("\n--- probe new token (single GET on small file) ---")
out, _ = run(
    f'curl -sS -o /tmp/_nt.bin '
    f'-w "http_code=%{{http_code}} size=%{{size_download}}\\n" '
    f'-D /tmp/_nh.txt '
    f'-H "Authorization: Bearer {NEW}" '
    f'"https://www.kaggle.com/api/v1/competitions/data/download/'
    f'rogii-wellbore-geology-prediction/sample_submission.csv"'
)
print(out)

print("\n--- response headers (status + retry-after) ---")
out, _ = run('head -1 /tmp/_nh.txt; grep -iE "^retry-after" /tmp/_nh.txt 2>/dev/null || echo no_retry_after_header')
print(out)

print("\n--- file landed? ---")
out, _ = run("ls -la /tmp/_nt.bin; echo ---; head -3 /tmp/_nt.bin 2>/dev/null")
print(out)

cli.close()
