```
conda create -n IMAGE python=3.9.21
conda activate IMAGE
pip3 install torch torchvision torchaudio
pip install -e .
pip install -r requirements.txt
pip install -e ../loralib/
```

When install tokenizers, there may be an error like this:
```
transformers 4.4.2 requires tokenizers<0.11,>=0.10.1, but you have tokenizers 0.21.0 which is incompatible.
```
Ignore it, because the latest version of tokenizers is not compatible with the transformers 4.4.2.