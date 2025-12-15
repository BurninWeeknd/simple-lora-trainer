Requirements Overview
Only supports linux (Ubuntu 24.04 LTS), may or may not work on other OS types. 

1. Python
- Python 3.10 or newer recommended

2. GPU & CUDA
- NVIDIA GPU required for training
- CUDA version compatible with PyTorch

Optional (recommended):
- xformers (must match your PyTorch + CUDA version)

3. Training Backend (sd-scripts)
This project uses `sd-scripts` as the training backend.

You must install its dependencies manually:

- bash
cd trainer/sd-scripts
pip install -r requirements.txt
pip install accelerate
accelerate config

This app is made for simple training.
I tried to ignore the majority of settings that are hardly used, for a cleaner minimal UI that is understandable.
Trains only SD1.5 and SDXL, it will not support other models. 
Bug fixes if needed will be made. Otherwise this project is final.

(I recommend the use of conda env)

Small instructions manual below
$ cd ~/Lora_Trainer
$ pip install -r requirements.txt (Make sure you install sd-scripts requirements)
$ python app.py

In the UI - Name - then create 
Put your data set image !folder! inside of database folder from the newly created project. (This will depend on OS) path defaults to /home/yourusername
Do not put images directly inside dataset folder!

Configure and click train. Some warnings and hard stop warning are in place.






