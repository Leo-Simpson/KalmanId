import subprocess
from os.path import join, dirname
main_dir = dirname(__file__)
script_dir = join(main_dir, 'scripts')

files = [
    "example1.py",
    "example2_compute.py",
    "example2_plot.py",
    "example3_compute.py",
    "example3_plot.py",
    "example3_robustness.py"
]
for f in files:
    print(f"Running {f}...")
    subprocess.run(["python", join(script_dir, f)], check=True)
