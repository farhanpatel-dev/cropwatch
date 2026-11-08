# ADR 0003: Limit uploads to 6000 x 6000 pixels

- Status: Proposed (needs Ibrahim's agreement: it changes docs/api.md and affects the gateway and frontend)
- Date: 2026-11-08
- Owner: Farhan (ML / inference)

## Context
The contract allowed images up to 8000 x 8000 pixels. The gateway calls inference `POST /predict` with a 60 s timeout (docs/architecture.md). Inference runs on CPU (ADR 0001) and its time grows with the number of pixels.

Measured on the dev laptop (16 logical CPUs, ONNX Runtime, full `/predict` request including decoding and polygons):

| Image | Time | Where |
| --- | --- | --- |
| 512 x 512 | 0.8 s | Docker container |
| 4096 x 4096 | 19.8 s | local server |
| 6000 x 6000 | 40.2 s and 41.0 s | Docker container |
| 8000 x 8000 | about 75 s (estimate from the above, not measured) | |

Tuning the batch size and number of threads saved about 25%; the service now uses batches of 16 tiles. That is not enough to bring 8000 px under 60 s.

## Decision
The inference service rejects images larger than 6000 x 6000 pixels with HTTP 413. `docs/api.md` is updated in the same pull request as the code (`services/inference/app/config.py`, `MAX_PIXELS_SIDE`).

## Alternatives considered
- **Raise the gateway timeout to 120 s.** Keeps 8000 px but users wait over a minute with no progress, and a stuck request holds resources twice as long.
- **Asynchronous jobs (upload returns a job id, the browser polls).** The right design at larger scale, but it needs a queue and a job table: too much for the 4-day MVP.
- **Downscale large images before inference.** Fast, but the model was trained at the dataset's native resolution, so results on resized images are untested and likely worse.
- **GPU inference.** Costs money (ADR 0001).

## Consequences
- Good: the largest allowed image finishes in about 41 s on the dev laptop, about 19 s under the timeout.
- Cost: users with bigger images must crop or split them first. The 413 error message says the limit.
- Risk: on a host with fewer CPUs, 6000 px may exceed 60 s. Re-measure on any new host and lower the limit (for example to 5000) if needed.
- Memory: the container used 1.2 GB after 6000 px requests. A free host with 512 MB RAM (for example Render's free plan, checked 2026-11-03 in docs/architecture.md) would need a much smaller limit. The default plan is a recorded demo of the local stack, so this does not block the MVP.
- The gateway and frontend should check the same limit, so users get the error before uploading 50 MB.
