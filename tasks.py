"""Cross-platform task runner (use instead of the Makefile on Windows).

    python tasks.py setup | download | prepare | train | evaluate | app | test | all
"""
import subprocess
import sys

PY = sys.executable

TASKS = {
    "setup": [[PY, "-m", "pip", "install", "-r", "requirements.txt"],
              [PY, "-m", "pip", "install", "-e", "."],
              [PY, "-m", "spacy", "download", "en_core_web_sm"]],
    "download": [[PY, "scripts/download_data.py"]],
    "prepare": [[PY, "scripts/prepare_data.py"]],
    "train": [[PY, "-m", "stylometry.models.train"]],
    "evaluate": [[PY, "scripts/run_experiments.py"],
                 [PY, "scripts/make_report_tables.py"]],
    "app": [[PY, "-m", "streamlit", "run", "app/streamlit_app.py"]],
    "test": [[PY, "-m", "pytest", "-q"]],
}
TASKS["all"] = TASKS["download"] + TASKS["prepare"] + TASKS["train"] + TASKS["evaluate"] + TASKS["test"]


def main(names):
    if not names or any(n not in TASKS for n in names):
        print(__doc__)
        sys.exit(1)
    for name in names:
        for cmd in TASKS[name]:
            print("$", " ".join(cmd))
            code = subprocess.call(cmd)
            if code != 0:
                sys.exit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
