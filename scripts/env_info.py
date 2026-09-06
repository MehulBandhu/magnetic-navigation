"""Print the environment a result was produced in. Run it next to any experiment and paste the
output into the commit message or the write-up."""
import platform
import subprocess

import numpy, scipy, matplotlib, torch

commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"commit {commit or 'n/a'}")
print(f"python {platform.python_version()}  torch {torch.__version__}  numpy {numpy.__version__}  scipy {scipy.__version__}  matplotlib {matplotlib.__version__}")
if torch.cuda.is_available():
    print(f"gpu {torch.cuda.get_device_name()}  cuda {torch.version.cuda}  bf16 {torch.cuda.is_bf16_supported(including_emulation=False)}")
else:
    print("gpu none (cpu)")
