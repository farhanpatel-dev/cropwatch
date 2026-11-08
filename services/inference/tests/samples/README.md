# Sample images for testing

Five small images for trying the inference service, the gateway and the frontend by hand. They are not used by the automated tests.

| File | Ground truth (share of field pixels) | What v3 returns |
| --- | --- | --- |
| `nutrient_deficiency.jpg` | 84% nutrient_deficiency | 1 nutrient_deficiency polygon |
| `drydown.jpg` | 94% drydown | 1 drydown polygon |
| `weed_cluster.jpg` | 43% weed_cluster | 3-4 weed_cluster polygons (one large, a few tiny) |
| `healthy.jpg` | 100% background | no polygons (empty `features`) |
| `weed_cluster_utm.tif` | same pixels as `weed_cluster.jpg` | 3 weed_cluster polygons, about 1,500 m2 in total |

These images were picked because the model does well on them: among about 25 candidates per class, the one with the best IoU. They show that the pipeline works; they do not show how accurate the model is. On average v3 misses about a third of nutrient-deficiency pixels and almost half of weed pixels, and marks 29.5% of healthy ground as stress (see `docs/model-card.md`).

## Using them
The JPEGs carry no location, so `bounds` (minLon,minLat,maxLon,maxLat) is required. Any small box works; this one is made up:
```powershell
curl.exe -F "file=@services/inference/tests/samples/drydown.jpg" -F "bounds=-93.62,41.99,-93.619,41.991" http://localhost:8001/predict
```
`weed_cluster_utm.tif` is a GeoTIFF with its own (made-up) location: UTM zone 15N (EPSG:32615), 0.1 m pixels, top-left corner at easting 450000, northing 4650000. Send it without `bounds`; the response's `image.bounds` is about `[-93.6037, 41.9999, -93.6031, 42.0004]`.
```powershell
curl.exe -F "file=@services/inference/tests/samples/weed_cluster_utm.tif" http://localhost:8001/predict
```
Areas depend on the bounds you give: the same image placed on a bigger box gives bigger `area_m2`.

Small differences are normal: JPEG decoding differs slightly between Windows and the Linux container, which can add or remove a polygon of a few m2. The GeoTIFF is lossless and gives the same result everywhere.

## Source
The four JPEGs are unchanged 512 x 512 images from the Agriculture-Vision 2021 validation split (fields 186RDEZTE, I29HUMXXF, NAR19ICN1, GE98HMMPY). Their use is subject to the dataset's terms: https://github.com/SHI-Labs/Agriculture-Vision. The GeoTIFF was made from `weed_cluster.jpg` with an invented georeference.
