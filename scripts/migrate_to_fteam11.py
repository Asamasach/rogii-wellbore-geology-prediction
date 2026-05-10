"""Full migration: fteam6 -> fteam11 via LAN.

1. Create rogii project dirs on fteam11 (via jump)
2. Install sshpass on fteam6 (needed for scp w/ password to fteam11)
3. SCP rogii.zip from fteam6 to fteam11 over LAN (fast)
4. Unzip on fteam11
5. Install Python data stack on fteam11 via pip --user
6. Upload our local scripts to fteam11 via SFTP-through-jump
7. Verify ready to run
"""
import os
import sys
import paramiko
from pathlib import Path

J_HOST = os.environ["FTEAM6_HOST"]; J_PORT = int(os.environ["FTEAM6_PORT"])
J_USER = os.environ["FTEAM6_USER"]; J_PASS = os.environ["FTEAM6_PASS"]
T_HOST = os.environ["FTEAM11_HOST"]; T_PORT = int(os.environ["FTEAM11_PORT"])
T_USER = os.environ["FTEAM11_USER"]; T_PASS = os.environ["FTEAM11_PASS"]

REMOTE_TARGET = "/home/fteam11/projects/rogii-wellbore-geology-prediction"
ZIP_SOURCE = "/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw/rogii.zip"


def ssh_to_fteam6():
    j = paramiko.SSHClient()
    j.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    j.connect(J_HOST, port=J_PORT, username=J_USER, password=J_PASS, timeout=15)
    return j


def ssh_to_fteam11_via_jump():
    j = ssh_to_fteam6()
    chan = j.get_transport().open_channel("direct-tcpip", (T_HOST, T_PORT), ("127.0.0.1", 0))
    t = paramiko.SSHClient()
    t.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    t.connect(T_HOST, port=T_PORT, username=T_USER, password=T_PASS, timeout=15, sock=chan)
    return j, t


def run(cli, c, t=600, label=""):
    if label: print(f"--- {label} ---")
    print(f"$ {c[:200]}{'...' if len(c) > 200 else ''}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t, get_pty=False)
    for ln in iter(o.readline, ""):
        if not ln: break
        try: print(ln.rstrip(), flush=True)
        except UnicodeEncodeError: print(ln.encode("ascii", "replace").decode().rstrip(), flush=True)
    err = e.read().decode(errors="replace").rstrip()
    if err:
        try: print(f"[stderr] {err}", flush=True)
        except UnicodeEncodeError: print("[stderr]", err.encode("ascii", "replace").decode())
    rc = o.channel.recv_exit_status()
    print(f"rc={rc}\n", flush=True)
    return rc


# Step 1: dirs on fteam11
print("=" * 70)
print("Step 1: create rogii dirs on fteam11")
print("=" * 70)
j, t11 = ssh_to_fteam11_via_jump()
run(t11,
    f"mkdir -p {REMOTE_TARGET}/data/raw {REMOTE_TARGET}/data/features "
    f"{REMOTE_TARGET}/scripts {REMOTE_TARGET}/models/oof "
    f"{REMOTE_TARGET}/models/pte {REMOTE_TARGET}/submissions/archived "
    f"{REMOTE_TARGET}/logs {REMOTE_TARGET}/public_baselines && "
    f"ls -la {REMOTE_TARGET}/")
t11.close(); j.close()

# Step 2: install sshpass on fteam6 (so we can scp w/ password from fteam6 to fteam11)
print("=" * 70)
print("Step 2: install sshpass on fteam6")
print("=" * 70)
f6 = ssh_to_fteam6()
run(f6,
    f"command -v sshpass >/dev/null && echo already_installed || "
    f"(echo '{J_PASS}' | sudo -S apt-get install -y sshpass 2>&1 | tail -5)")

# Step 3: scp rogii.zip from fteam6 -> fteam11 over LAN
print("=" * 70)
print("Step 3: scp rogii.zip fteam6 -> fteam11 (LAN)")
print("=" * 70)
run(f6,
    f"sshpass -p '{T_PASS}' scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
    f"{ZIP_SOURCE} {T_USER}@{T_HOST}:{REMOTE_TARGET}/data/raw/rogii.zip 2>&1; "
    f"echo done",
    t=1800)
f6.close()

# Step 4: unzip + install Python deps on fteam11
print("=" * 70)
print("Step 4: unzip + verify on fteam11")
print("=" * 70)
j, t11 = ssh_to_fteam11_via_jump()
run(t11,
    f"cd {REMOTE_TARGET}/data/raw && "
    f"ls -la rogii.zip && "
    f"unzip -q -o rogii.zip && "
    f"echo files=$(find . -type f ! -name rogii.zip | wc -l) "
    f"size_mb=$(du -sm --exclude=rogii.zip . | cut -f1)",
    t=600)

# Step 5: pip install --user (pip is already on fteam11)
print("=" * 70)
print("Step 5: pip install numpy/pandas/scipy/sklearn on fteam11")
print("=" * 70)
run(t11,
    "python3 -c 'import numpy, pandas, scipy, sklearn' 2>/dev/null && echo ALREADY_HAVE || "
    "pip3 install --user --quiet numpy pandas scipy scikit-learn lightgbm 2>&1 | tail -5",
    t=600)
run(t11,
    'python3 -c "import numpy,pandas,scipy,sklearn; '
    'print(\\"numpy\\",numpy.__version__,\\"pandas\\",pandas.__version__,'
    '\\"scipy\\",scipy.__version__,\\"sklearn\\",sklearn.__version__)"',
    t=30)
t11.close(); j.close()

# Step 6: upload our local scripts via SFTP-through-jump
print("=" * 70)
print("Step 6: upload local scripts to fteam11")
print("=" * 70)
j = ssh_to_fteam6()
chan = j.get_transport().open_channel("direct-tcpip", (T_HOST, T_PORT), ("127.0.0.1", 0))
t11 = paramiko.SSHClient()
t11.set_missing_host_key_policy(paramiko.AutoAddPolicy())
t11.connect(T_HOST, port=T_PORT, username=T_USER, password=T_PASS, timeout=15, sock=chan)
sftp = t11.open_sftp()

LOCAL_SCRIPTS = Path(r"c:\projects\kaggle\rogii-wellbore-geology-prediction\scripts")
to_upload = [
    "predict_tvt.py",
    "eval_baselines_remote.py",
    "make_submission_const.py",
]
try:
    for fname in to_upload:
        local = LOCAL_SCRIPTS / fname
        if not local.exists():
            print(f"  skip (missing local): {fname}")
            continue
        remote = f"{REMOTE_TARGET}/scripts/{fname}"
        sftp.put(str(local), remote)
        print(f"  uploaded {fname} ({local.stat().st_size} bytes)")
finally:
    sftp.close()

# Step 7: smoke-test by running the baseline eval
print("=" * 70)
print("Step 7: smoke-test — re-run baseline eval on fteam11")
print("=" * 70)
# Patch ROOT path inside eval script's reference: it uses /home/fteam6/...
# Easier: just invoke with cwd; the script computes paths from __file__.
# eval_baselines_remote.py has hardcoded "/home/fteam6/..." -> it WILL fail.
# Fix it inline by sed-ing the path.
run(t11,
    f"sed -i 's|/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw|"
    f"{REMOTE_TARGET}/data/raw|g' {REMOTE_TARGET}/scripts/eval_baselines_remote.py && "
    f"head -20 {REMOTE_TARGET}/scripts/eval_baselines_remote.py | grep -i ROOT")

run(t11,
    f"cd {REMOTE_TARGET} && python3 -u scripts/eval_baselines_remote.py 2>&1 | tee logs/baseline_eval_fteam11.log",
    t=2400)

t11.close(); j.close()
print("\n[migration done]")
