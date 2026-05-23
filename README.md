# SwishNet

PyTorch implementation of the SwishNet architecture.

[arXiv:1812.00149: SwishNet: A Fast Convolutional Neural Network for Speech, Music and Noise Classification and Segmentation](https://arxiv.org/abs/1812.00149).

# Usage

```python
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# MFCC frames as input: (batch_size, num_frames, num_coefficients)
x = torch.randn((1, 20, 32)).to(device)

model = SwishNet(in_channels=20, out_channels=2).to(device)
model(x)
```

In order to load a real audio file you can use torchaudio:

```python
waveform, sample_rate = torchaudio.load("test.wav", normalize=True)
transform = transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=20,
)

mfccs = transform(waveform).to(device)
model(mfccs)
```

# Development environment

```bash
uv sync --dev
source .venv/bin/activate
```

# Tests

```bash
uv run pytest
```

If running on WSL:

```bash
cd $(uv pip show torch | grep Location | awk -F ": " '{print $2}')/torch/lib
rm libhsa-runtime64.so*
cp /opt/rocm/lib/libhsa-runtime64.so.1.* libhsa-runtime64.so
```