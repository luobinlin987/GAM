import subprocess
import sys
import os

scripts = ["5_2_1forward_verification_harmonic_seepage.py", "5_2_2forward_verification_biharmonic_elasticity.py"]
#scripts = ["5_2_1forward_verification_harmonic_seepage.py"]
#scripts = ["5_2_2forward_verification_biharmonic_elasticity.py"]


env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"

with open("results.md", "w", encoding="utf-8") as f:
    for script in scripts:
        print(f"\n{'='*40}\nRunning: {script}\n{'='*40}")

        f.write(f"# {script}\n\n")
        f.write("```\n")
        f.flush()

        process = subprocess.Popen(
            [sys.executable, "-u", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env
        )

        for line in process.stdout:
            print(line, end='')
            f.write(line)
            f.flush()

        process.wait()

        f.write("\n```\n\n")
        f.flush()

print("\nComputation complete, all outputs are recorded in file results.md")
