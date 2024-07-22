import os
import sys
sys.path.append(".")
sys.path.append("./nerfstudio")

from nerfstudio.scripts.viewer.run_viewer import entrypoint

if __name__ == "__main__":
    entrypoint()