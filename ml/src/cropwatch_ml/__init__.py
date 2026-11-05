"""CropWatch ML: datasets, models and training for crop-stress segmentation.

torch is imported first on purpose: on Windows, importing mlflow, pandas or pyarrow
before torch fails with WinError 1114 (pyarrow 17 ships an older msvcp140.dll).
"""

import torch  # noqa: F401  (must stay the first import)
