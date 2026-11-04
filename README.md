# cropwatch
Geospatial crop-health monitoring platform. A PyTorch U-Net segments crop-stress regions in aerial and satellite imagery, served through a FastAPI service that returns georeferenced polygons stored in PostGIS. Dockerized, tracked with MLflow, CI/CD via GitHub Actions, deployed on Azure.
