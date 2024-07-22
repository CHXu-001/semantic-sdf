import ensurepip
import pathlib
import subprocess
import sys

ensurepip.bootstrap()
pybin = sys.executable

current_folder = pathlib.Path(__file__)
subprocess.check_call([pybin, "-m", "pip", "install", "-e", current_folder.parents[2]])
