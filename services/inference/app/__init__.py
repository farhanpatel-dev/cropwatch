"""Inference service package.

rasterio ships its own PROJ and GDAL data files that match its version. Installers such as
PostgreSQL/PostGIS on Windows set PROJ_LIB / GDAL_DATA machine-wide to their own, older
copies, and rasterio then fails with "proj.db ... comes from another PROJ installation".
We drop those variables for this process only (before rasterio is imported), so the
bundled data is used. The Docker image does not set them, so this changes nothing there.
"""

import os

for _var in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_var, None)
