import sys

import torch
import transformers
import pandas as pd
import sklearn

print("=" * 50)
print("C1 HGSS-ABSA Environment Check")
print("=" * 50)

print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU: No CUDA GPU detected")