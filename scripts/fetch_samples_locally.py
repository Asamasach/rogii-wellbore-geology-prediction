"""SCP the sample submission and a handful of typewell CSVs from fteam6 to local."""
import os
import paramiko
from pathlib import Path

HOST = os.environ["FTEAM6_HOST"]; PORT = int(os.environ["FTEAM6_PORT"])
USER = os.environ["FTEAM6_USER"]; PASS = os.environ["FTEAM6_PASS"]

REMOTE = "/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw"
LOCAL = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw")
LOCAL.mkdir(parents=True, exist_ok=True)
(LOCAL / "_samples_train").mkdir(exist_ok=True)
(LOCAL / "_samples_test").mkdir(exist_ok=True)

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=20)

# 1) sample_submission.csv — pull from /tmp where the smoke test wrote it (or
#    from data/raw if it landed there in the relaunch).
def cmd(c, t=30):
    stdin, stdout, _ = cli.exec_command(c, timeout=t)
    return stdout.read().decode(errors="replace").strip()

ss_remote = "/tmp/_ss.csv"
ss_size = cmd(f"ls -la {ss_remote} 2>/dev/null | awk '{{print $5}}'")
print(f"sample_submission on fteam6: {ss_size} bytes")

sftp = cli.open_sftp()
try:
    sftp.get(ss_remote, str(LOCAL / "sample_submission.csv"))
    print(f"  pulled to {LOCAL / 'sample_submission.csv'}")

    # 2) Get list of typewell files we have
    train_files = cmd(f"find {REMOTE}/train -name '*typewell.csv' 2>/dev/null | head -8").splitlines()
    test_files = cmd(f"find {REMOTE}/test -type f 2>/dev/null").splitlines()
    print(f"\n  train typewell files available: pulling {len(train_files)} samples")
    for r in train_files:
        if not r:
            continue
        local = LOCAL / "_samples_train" / Path(r).name
        sftp.get(r, str(local))
        print(f"    {Path(r).name}  ({local.stat().st_size} bytes)")
    print(f"\n  test files available: {len(test_files)}")
    for r in test_files:
        if not r:
            continue
        local = LOCAL / "_samples_test" / Path(r).name
        sftp.get(r, str(local))
        print(f"    {Path(r).name}  ({local.stat().st_size} bytes)")
finally:
    sftp.close()
    cli.close()
