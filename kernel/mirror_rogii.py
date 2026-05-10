"""Zip mounted comp data into /kaggle/working/rogii.zip — auto-detects mount path."""
import os
import shutil
import sys

print("=== /kaggle/input tree (depth 2) ===", flush=True)
for entry in sorted(os.listdir("/kaggle/input")):
    p1 = f"/kaggle/input/{entry}"
    print(f"  /kaggle/input/{entry}", flush=True)
    if os.path.isdir(p1):
        for sub in sorted(os.listdir(p1))[:10]:
            p2 = os.path.join(p1, sub)
            n = (sum(len(f) for _, _, f in os.walk(p2)) if os.path.isdir(p2) else 1)
            print(f"    {sub}  ({n} items)", flush=True)

# Pick the directory containing the comp data — look for typewell.csv files
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if any(f.endswith("typewell.csv") for f in files) or "typewell.csv" in str(files):
        # Climb up to the actual root that contains both train/ and test/
        if "train" in root or "test" in root:
            src = os.path.dirname(root)
            break
        else:
            src = root
            break

if src is None:
    sys.exit("error: could not find rogii data root in /kaggle/input")

print(f"\n=== chose src={src} ===", flush=True)
n = sum(len(f) for _, _, f in os.walk(src))
sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(src) for f in fs) / 1e6
print(f"files={n} size_mb={sz:.1f}", flush=True)

print(f"\n=== zipping to /kaggle/working/rogii.zip ===", flush=True)
shutil.make_archive("/kaggle/working/rogii", "zip", src)
zsz = os.path.getsize("/kaggle/working/rogii.zip") / 1e6
print(f"zip done: {zsz:.1f} MB", flush=True)
