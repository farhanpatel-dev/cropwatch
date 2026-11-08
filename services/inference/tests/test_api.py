from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from app.main import create_app
from app.model import ModelInfo, Segmenter

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
BOUNDS = "-123.2,49.1,-123.19,49.11"


def fake_info() -> ModelInfo:
    names = ["background", "nutrient_deficiency", "drydown", "weed_cluster"]
    return ModelInfo(
        name="fake",
        version="9",
        mlflow_run_id="run123",
        tile_size=256,
        input_channels=3,
        classes=[{"id": i, "name": n} for i, n in enumerate(names)],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        input_name="image",
        output_name="logits",
    )


class RedIsWeeds:
    """Fake model: pixels with red > 0 (normalised > -1) are weed_cluster, the rest background."""

    def run(self, output_names, feed):
        red = feed["image"][:, 0]
        logits = np.zeros((red.shape[0], 4, *red.shape[1:]), np.float32)
        logits[:, 0] = 1.0
        logits[:, 3] = np.where(red > -0.99, 3.0, 0.0)
        return [logits]


@pytest.fixture
def client():
    with TestClient(create_app(Segmenter(fake_info(), RedIsWeeds()))) as c:
        yield c


def encode(array: np.ndarray, driver: str, **profile) -> bytes:
    """bands x H x W array -> image file bytes."""
    count, h, w = array.shape
    with MemoryFile() as mem:
        with mem.open(
            driver=driver, width=w, height=h, count=count, dtype=array.dtype, **profile
        ) as dst:
            dst.write(array)
        return mem.read()


def field_png(h: int = 300, w: int = 400) -> bytes:
    img = np.zeros((3, h, w), np.uint8)
    img[0, 50:150, 100:200] = 200  # one 100 x 100 "weed" square
    return encode(img, "PNG")


def post(client: TestClient, data: bytes, bounds: str | None = BOUNDS, name: str = "f.png"):
    form = {"bounds": bounds} if bounds is not None else {}
    return client.post("/predict", files={"file": (name, data)}, data=form)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_model_info(client: TestClient) -> None:
    body = client.get("/model-info").json()
    assert body["name"] == "fake" and body["version"] == "9"
    assert body["mlflow_run_id"] == "run123"
    assert body["tile_size"] == 256 and body["input_channels"] == 3
    assert body["classes"][3] == {"id": 3, "name": "weed_cluster"}
    assert body["preprocessing"] == {"mean": [0.5] * 3, "std": [0.5] * 3}


def test_predict_png_with_bounds(client: TestClient) -> None:
    r = post(client, field_png())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == {"name": "fake", "version": "9"}
    assert body["image"]["width"] == 400 and body["image"]["height"] == 300
    assert body["image"]["crs"] == "EPSG:4326"
    assert body["image"]["bounds"] == pytest.approx([-123.2, 49.1, -123.19, 49.11])
    (f,) = body["polygons"]["features"]
    assert f["geometry"]["type"] == "MultiPolygon"
    assert f["properties"]["class_name"] == "weed_cluster"
    # image covers about 729 m x 1112 m = 810,000 m2; the square is 1/12 of it
    assert f["properties"]["area_m2"] == pytest.approx(810_000 / 12, rel=0.01)


def test_geotiff_uses_its_own_crs_and_ignores_bounds(client: TestClient) -> None:
    img = np.zeros((3, 256, 256), np.uint8)
    img[0, :100, :100] = 255
    utm = Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5440000.0)  # 1 m pixels near -123, 49.1
    data = encode(img, "GTiff", crs="EPSG:32610", transform=utm)
    r = post(client, data, bounds="0,0,1,1", name="f.tif")
    assert r.status_code == 200, r.text
    lon0, lat0, _, _ = r.json()["image"]["bounds"]
    assert lon0 == pytest.approx(-123.0, abs=1e-6) and 49.0 < lat0 < 49.2
    (f,) = r.json()["polygons"]["features"]
    assert f["properties"]["area_m2"] == pytest.approx(10_000, rel=0.002)


def test_extra_bands_ignored(client: TestClient) -> None:
    rgba = np.zeros((4, 64, 64), np.uint8)
    rgba[3] = 255  # alpha channel
    r = post(client, encode(rgba, "PNG"))
    assert r.status_code == 200 and r.json()["polygons"]["features"] == []


@pytest.mark.parametrize(
    ("data", "bounds", "status", "message"),
    [
        (b"not an image", BOUNDS, 400, "not a readable image"),
        (b"", BOUNDS, 400, "empty"),
        (None, None, 400, "bounds are required"),
        (None, "1,2,3", 400, "four numbers"),
        (None, "a,b,c,d", 400, "four numbers"),
        (None, "-123.1,49.1,-123.2,49.2", 400, "min < max"),
    ],
)
def test_bad_requests(client: TestClient, data, bounds, status, message) -> None:
    r = post(client, field_png() if data is None else data, bounds=bounds)
    assert r.status_code == status
    assert message in r.json()["detail"]


def test_single_band_rejected(client: TestClient) -> None:
    r = post(client, encode(np.zeros((1, 32, 32), np.uint8), "PNG"))
    assert r.status_code == 400 and "band" in r.json()["detail"]


def test_16_bit_rejected(client: TestClient) -> None:
    r = post(client, encode(np.zeros((3, 32, 32), np.uint16), "PNG"))
    assert r.status_code == 400 and "8-bit" in r.json()["detail"]


def test_too_large_rejected(client: TestClient) -> None:
    r = post(client, encode(np.zeros((3, 2, 6001), np.uint8), "PNG"))
    assert r.status_code == 413
    assert post(client, encode(np.zeros((3, 2, 6000), np.uint8), "PNG")).status_code == 200


def test_request_id_reused_or_created(client: TestClient) -> None:
    assert client.get("/health", headers={"X-Request-ID": "abc"}).headers["X-Request-ID"] == "abc"
    assert len(client.get("/health").headers["X-Request-ID"]) == 32


@pytest.mark.skipif(not (MODEL_DIR / "model.onnx").exists(), reason="model.onnx not present")
def test_real_model_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", str(MODEL_DIR / "model.onnx"))
    monkeypatch.setenv("MODEL_INFO_PATH", str(MODEL_DIR / "model_info.json"))
    with TestClient(create_app()) as c:
        version = c.get("/model-info").json()["version"]
        r = post(c, field_png(512, 512))
    assert r.status_code == 200, r.text
    assert r.json()["model"]["version"] == version
    for f in r.json()["polygons"]["features"]:
        assert 1 <= f["properties"]["class_id"] <= 3
        assert 0.25 <= f["properties"]["confidence"] <= 1.0
