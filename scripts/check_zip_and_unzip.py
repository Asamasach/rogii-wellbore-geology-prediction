"""Check whether rogii.zip landed on fteam6 (despite earlier print crash) and unzip if so."""
import os
import sys
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


def run(c, t=600):
    _, o, e = cli.exec_command(c, timeout=t)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    rc = o.channel.recv_exit_status()
    # Sanitise output of any Chinese chars before printing under cp1252
    safe = lambda s: s.encode("ascii", "replace").decode()
    print(f"$ {c[:160]}")
    if out: print(safe(out)[:1500])
    if err: print(f"[stderr] {safe(err)[:500]}")
    print(f"rc={rc}\n", flush=True)
    return rc, out


print("=== rogii.zip on fteam6? ===")
run(f"stat -c '%n size=%s mtime=%y' {R}/rogii.zip 2>&1")

print("=== quick zip integrity check ===")
run(f"unzip -l {R}/rogii.zip 2>&1 | head -3 && echo ... && unzip -l {R}/rogii.zip 2>&1 | tail -3")

print("=== unzipping (in place into data/raw, replacing the partial earlier files) ===")
# Safest: unzip into a fresh subdir, then move into data/raw root
run(
    f"cd {R} && rm -rf _extract && mkdir _extract && "
    f"unzip -q rogii.zip -d _extract && "
    f"echo extracted_files=$(find _extract -type f | wc -l) "
    f"size_mb=$(du -sm _extract | cut -f1)",
    t=600,
)

print("=== reorganise into data/raw/{train,test,*} ===")
# After extraction, _extract/ contains: train/, test/, sample_submission.csv,
# AI_*.pptx, etc. (the comp data root). Move everything one level up,
# overwriting any partial files we already had.
run(
    f"cd {R} && "
    f"cp -rf _extract/* . && "
    f"echo train_typewells=$(find train -name '*typewell.csv' | wc -l) "
    f"train_horizontals=$(find train -name '*horizontal_well.csv' | wc -l) "
    f"test_files=$(find test -type f | wc -l) "
    f"pptx=$(ls AI_*.pptx 2>/dev/null | wc -l) "
    f"sample_sub=$(ls sample_submission.csv 2>/dev/null | wc -l)",
    t=300,
)

print("=== final disk state ===")
run(f"echo total_files=$(find {R} -type f ! -path '*/_extract/*' ! -name rogii.zip | wc -l) "
    f"total_size_mb=$(du -sm --exclude=_extract --exclude=rogii.zip {R} | cut -f1)")

cli.close()
