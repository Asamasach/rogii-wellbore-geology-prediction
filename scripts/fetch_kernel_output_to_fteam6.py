"""Download the kernel output zip directly to fteam6, then unzip in place."""
import json
import os
import paramiko

TOKEN = os.environ["KAGGLE_API_TOKEN"]
USER = "asamasach"
KSLUG = "rogii-mirror"
REMOTE_RAW = "/home/fteam6/project/rogii-wellbore-geology-prediction/data/raw"

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect(
    os.environ["FTEAM6_HOST"],
    port=int(os.environ["FTEAM6_PORT"]),
    username=os.environ["FTEAM6_USER"],
    password=os.environ["FTEAM6_PASS"],
    timeout=15,
)


def run(c, t=600, capture_only=False):
    if not capture_only:
        print(f"$ {c[:200]}{'...' if len(c) > 200 else ''}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    rc = o.channel.recv_exit_status()
    if not capture_only:
        if out: print(out[:2000])
        if err: print(f"[stderr] {err[:500]}")
        print(f"rc={rc}\n", flush=True)
    return rc, out, err


# 1) Fetch kernel-output metadata (JSON listing of output files + URLs)
print("=== fetch output metadata ===")
url = f"https://www.kaggle.com/api/v1/kernels/output?userName={USER}&kernelSlug={KSLUG}"
rc, out, _ = run(
    f'curl -sS -H "Authorization: Bearer {TOKEN}" "{url}"',
    capture_only=True,
)
print(out[:1500])
print()

meta = json.loads(out)
files = meta.get("files", [])
print(f"output files: {len(files)}")
for f in files:
    print(f"  {f.get('fileName')!r}  size={f.get('fileSize')}  url={f.get('url')[:80] if f.get('url') else ''}...")

# 2) Find rogii.zip and download it via the presigned URL
target = next((f for f in files if "rogii.zip" in f.get("fileName", "")), None)
if not target:
    raise SystemExit("error: rogii.zip not in kernel output")

dl_url = target["url"]
fsize = target.get("fileSize", 0)
print(f"\n=== downloading {target['fileName']} ({fsize/1e6:.1f} MB) to fteam6 ===")
# Don't echo the long presigned URL fully — it has a signature
print(f"  (presigned URL prefix: {dl_url[:80]}...)")

# Use curl with -L (follow redirects), -C - (resume on retry if needed)
# The presigned URL doesn't need our auth header.
remote_zip = f"{REMOTE_RAW}/rogii.zip"
quoted = dl_url.replace('"', '\\"')
rc, _, _ = run(
    f'curl -sS -L --retry 3 --retry-delay 5 -o "{remote_zip}" "{quoted}" && '
    f'ls -la "{remote_zip}" && '
    f'echo md5: && md5sum "{remote_zip}" | cut -d" " -f1',
    t=2400,  # 40 min for an 800 MB download at 0.5+ MB/s
)

# 3) Unzip
print("=== unzip ===")
run(
    f'cd {REMOTE_RAW} && '
    f'rm -rf rogii_extracted && mkdir rogii_extracted && '
    f'unzip -q rogii.zip -d rogii_extracted && '
    f'echo extracted: $(find rogii_extracted -type f | wc -l) files, '
    f'$(du -sm rogii_extracted | cut -f1) MB',
    t=600,
)

cli.close()
