import subprocess
import sys
import os

scripts = ["5_3_1backward_verification_harmonic.py", "5_3_2backward_verification_biharmonic.py"]
#scripts = ["5_3_1backward_verification_harmonic.py"]
#scripts = ["5_3_2backward_verification_biharmonic.py"]


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
