```
conda create -n IMAGE python=3.9.21
conda activate IMAGE
pip3 install torch torchvision torchaudio
pip install -r requirements.txt
pip install -e .
pip install -e ../loralib/
```