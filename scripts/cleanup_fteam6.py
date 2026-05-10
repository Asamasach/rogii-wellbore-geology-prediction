"""Level-2 cleanup of fteam6: delete rogii project, remove kaggle_env, revert .bashrc."""
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


def run(c, t=300):
    print(f"$ {c}", flush=True)
    _, o, e = cli.exec_command(c, timeout=t)
    out = o.read().decode(errors="replace").strip()
    err = e.read().decode(errors="replace").strip()
    rc = o.channel.recv_exit_status()
    safe = lambda s: s.encode("ascii", "replace").decode()
    if out: print(safe(out))
    if err: print(f"[stderr] {safe(err)}")
    print(f"rc={rc}\n")


print("=== before: disk ===")
run("df -BG /home | tail -1; du -sh ~/project/ ~/.config/kaggle_env 2>/dev/null")

print("=== delete rogii project dir ===")
run("rm -rf ~/project/rogii-wellbore-geology-prediction && echo deleted; ls ~/project/ 2>&1")

print("=== remove kaggle_env file ===")
run("rm -f ~/.config/kaggle_env && echo deleted; ls -la ~/.config/kaggle_env 2>&1")

print("=== revert .bashrc line 164 (the kaggle_env source) ===")
# Use sed to delete the exact line we added
run('sed -i "/\\[ -f ~\\/.config\\/kaggle_env \\] && \\. ~\\/.config\\/kaggle_env/d" ~/.bashrc; '
    'grep -nE "kaggle|KAGGLE" ~/.bashrc || echo "no_kaggle_remaining"')

print("=== after: disk + final state ===")
run("df -BG /home | tail -1; du -sh ~/project/ 2>/dev/null; ls ~/project/")

cli.close()
print("[cleanup done]")
