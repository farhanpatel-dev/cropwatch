# ADR 0003: Limit uploads to 4000 x 4000 pixels

- Status: Accepted on 2026-11-20 with a 4000 px limit (Ibrahim). First proposed on 2026-11-08 with 6000 px.
- Date: 2026-11-08, revised 2026-11-20
- Owner: Farhan (ML / inference); revised by Ibrahim

## Context
The contract allowed images up to 8000 x 8000 pixels. The gateway calls inference `POST /predict` with a 60 s timeout (docs/architecture.md). Inference runs on CPU (ADR 0001) and its time grows with the number of pixels.

Measured on the dev laptop (16 logical CPUs, ONNX Runtime, full `/predict` request including decoding and polygons):

| Image | Time | Where |
| --- | --- | --- |
| 512 x 512 | 0.8 s | Docker container |
| 4096 x 4096 | 19.8 s | local server |
| 6000 x 6000 | 40.2 s and 41.0 s | Docker container |
| 8000 x 8000 | about 75 s (estimate from the above, not measured) | |

Re-measured on 2026-11-20 on Ibrahim's laptop (Apple M2, 8 cores; Docker in Colima with 6 CPUs and 5 GB):

| Image | Time | Where |
| --- | --- | --- |
| 512 x 512 | 0.7-1.2 s | gateway -> inference -> geo, Docker |
| 4000 x 4000 | 25.8 s and 31.1 s | gateway -> inference -> geo, Docker |
| 6000 x 6000 | 74.6 s (inference alone; the gateway returned 504) | Docker |
| 6000 x 6000 | 43.3 s | natively on macOS, no Docker, no HTTP |

Tuning the batch size and number of threads saved about 25%; the service now uses batches of 16 tiles. That is not enough to bring 8000 px under 60 s.

## Decision
The inference service rejects images larger than 4000 x 4000 pixels with HTTP 413 (first set to 6000 on 2026-11-08; lowered after the M2 measurement, which showed 6000 px times out on an 8-core machine). The frontend checks the same limit before uploading. `docs/api.md` is updated in the same pull request as the code (`services/inference/app/config.py`, `MAX_PIXELS_SIDE`).

## Alternatives considered
- **Raise the gateway timeout to 120 s.** Keeps 8000 px but users wait over a minute with no progress, and a stuck request holds resources twice as long.
- **Asynchronous jobs (upload returns a job id, the browser polls).** The right design at larger scale, but it needs a queue and a job table: too much for the 4-day MVP.
- **Downscale large images before inference.** Fast, but the model was trained at the dataset's native resolution, so results on resized images are untested and likely worse.
- **GPU inference.** Costs money (ADR 0001).

## Consequences
- Good: the largest allowed image finishes in about 26-31 s on an 8-core laptop in Docker, about 30 s under the timeout.
- Cost: users with bigger images must crop or split them first. The 413 error message says the limit.
- Risk: on a host with fewer CPUs (for example a 2 vCPU cloud container), even 4000 px may exceed 60 s. Re-measure on any new host and lower the limit further if needed.
- Memory: the container used 1.2 GB after 6000 px requests (less at 4000 px). A free host with 512 MB RAM (for example Render's free plan, checked 2026-11-03 in docs/architecture.md) would need a much smaller limit. The default plan is a recorded demo of the local stack, so this does not block the MVP.
- The gateway and frontend should check the same limit, so users get the error before uploading 50 MB.
