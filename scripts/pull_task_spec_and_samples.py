"""Pull the .pptx + a few horizontal_well samples + sample_submission to local."""
import os
from pathlib import Path
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
LOCAL = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\data\raw")
LOCAL.mkdir(parents=True, exist_ok=True)
(LOCAL / "_samples_train").mkdir(exist_ok=True)
(LOCAL / "_samples_test").mkdir(exist_ok=True)


def cmd(c, t=30):
    _, o, _ = cli.exec_command(c, timeout=t)
    return o.read().decode(errors="replace").strip()


sftp = cli.open_sftp()
try:
    # 1) .pptx (the task description) — 28.8 MB. This is the priority.
    pptx_remote = cmd(f"ls {R}/AI_*.pptx 2>/dev/null | head -1")
    if pptx_remote:
        local_pptx = LOCAL / Path(pptx_remote).name
        print(f"pulling {Path(pptx_remote).name} ({cmd(f'stat -c%s {pptx_remote}')} bytes)...")
        sftp.get(pptx_remote, str(local_pptx))
        print(f"  -> {local_pptx}  ({local_pptx.stat().st_size} bytes)")

    # 2) sample_submission.csv (overwrite — earlier we had a copy)
    print(f"\npulling sample_submission.csv...")
    sftp.get(f"{R}/sample_submission.csv", str(LOCAL / "sample_submission.csv"))
    print(f"  -> {LOCAL / 'sample_submission.csv'}")

    # 3) A couple of train horizontal_well + typewell pairs (different wells)
    train_files = cmd(f"find {R}/train -name '*horizontal_well.csv' | head -4").splitlines()
    print(f"\npulling {len(train_files)} train horizontal samples + their typewells...")
    for r in train_files:
        if not r:
            continue
        well_id = Path(r).name.split("__")[0]
        local_h = LOCAL / "_samples_train" / Path(r).name
        sftp.get(r, str(local_h))
        # matching typewell
        type_remote = f"{R}/train/{well_id}__typewell.csv"
        local_t = LOCAL / "_samples_train" / Path(type_remote).name
        sftp.get(type_remote, str(local_t))
        print(f"  {well_id}: horizontal={local_h.stat().st_size} typewell={local_t.stat().st_size}")

    # 4) All 6 test files
    test_files = cmd(f"find {R}/test -type f").splitlines()
    print(f"\npulling all {len(test_files)} test files...")
    for r in test_files:
        if not r:
            continue
        local = LOCAL / "_samples_test" / Path(r).name
        sftp.get(r, str(local))
        print(f"  {Path(r).name}: {local.stat().st_size} bytes")
finally:
    sftp.close()
    cli.close()
