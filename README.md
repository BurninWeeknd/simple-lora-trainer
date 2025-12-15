LoRA Trainer (SD 1.5 & SDXL)
A simple web UI for training LoRA models using sd-scripts.
Designed to avoid overwhelming settings and focus on the ones that actually matter.
Supports SD 1.5 and SDXL only
For Linux (created & tested on Ubuntu 24.04 LTS)
Minimal UI, safety checks included
No advanced / experimental features by default
This project is considered feature-complete.
Only bug fixes will be added.
----------------------------
System Requirements:
Linux (Ubuntu recommended)
NVIDIA GPU (required for training)
CUDA compatible with your PyTorch install
----------------------------
Python - 
Python 3.10+ recommended
----------------------------
Conda environment strongly suggested
----------------------------
Optional - 
xFormers (must match your PyTorch + CUDA versions)
----------------------------
Training Backend
This app uses sd-scripts.
You must install its dependencies manually:
cd trainer/sd-scripts
pip install -r requirements.txt
pip install accelerate
accelerate config
----------------------------
Installation
git clone 
cd path/to/Lora_Trainer
pip install -r requirements.txt
python app.py
DONE
----------------------------
Usage
Create a project
Place your image folders inside the project’s dataset/ directory
Do not place images directly in dataset/ — use subfolders
Configure settings
Click Train
----------------------------
The UI includes:
Warnings for risky settings
Hard stops for invalid configurations
