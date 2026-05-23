import os
os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

import torch
import torch.utils.benchmark as benchmark
from swishnet import SwishNet, SwishNetWide

N_MFCC = 20
TIME_FRAMES = 32

gpu = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else None
)

model_configs = [
    ("SwishNet",     SwishNet(in_channels=N_MFCC, out_channels=2)),
    ("SwishNetWide", SwishNetWide(classes=2)),
]

batch_sizes = [1, 8, 32]

results = []
for name, base_model in model_configs:
    for device in filter(None, ["cpu", gpu]):
        model = base_model.to(device).eval()
        for batch in batch_sizes:
            x = torch.randn(batch, N_MFCC, TIME_FRAMES, device=device)
            with torch.no_grad():
                for _ in range(3):
                    model(x)
            results.append(benchmark.Timer(
                stmt="model(x)",
                globals={"model": model, "x": x},
                label=name,
                sub_label=f"batch={batch}",
                description=device,
            ).blocked_autorange(min_run_time=1.0))

benchmark.Compare(results).print()
