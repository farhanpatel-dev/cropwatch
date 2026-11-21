"""Gateway tests with fake inference and geo services (httpx.MockTransport): no model, no DB."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

AID = "0b7c1d1e-6e1f-4c58-9a3e-2f3f0a6a9b11"
PREDICTION = {
    "model": {"name": "unet-resnet34", "version": "3"},
    "image": {
        "width": 512,
        "height": 512,
        "crs": "EPSG:4326",
        "bounds": [-93.62, 41.99, -93.6, 42.0],
    },
    "polygons": {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": []},
                "properties": {
                    "class_id": 2,
                    "class_name": "drydown",
                    "confidence": 0.9,
                    "area_m2": 10.0,
                },
            }
        ],
    },
}
ROW = {
    "id": AID,
    "name": "north",
    "source_filename": "f.jpg",
    "model_name": "unet-resnet34",
    "model_version": "3",
    "status": "completed",
    "image_bounds": {"type": "Polygon", "coordinates": []},
    "created_at": "2026-11-20T10:00:00+00:00",
}
STATS = [{"class_id": 2, "class_name": "drydown", "polygon_count": 1, "area_ha": 0.001}]


class Fake:
    """Records requests; `inference` decides the /predict answer."""

    def __init__(self, inference=None):
        self.calls: list[httpx.Request] = []
        self.inference = inference or (lambda req: httpx.Response(200, json=PREDICTION))

    def __call__(self, req: httpx.Request) -> httpx.Response:
        self.calls.append(req)
        path = req.url.path
        if req.url.host == "inference" and path == "/model-info":
            return httpx.Response(200, json={"name": "unet-resnet34", "version": "3"})
        if req.url.host == "inference":
            return self.inference(req)
        if path == "/analyses" and req.method == "POST":
            return httpx.Response(201, json=ROW)
        if path == f"/analyses/{AID}/polygons" and req.method == "POST":
            return httpx.Response(201, json={"inserted": 1})
        if path == "/stats":
            return httpx.Response(200, json=STATS)
        if path == f"/analyses/{AID}":
            return httpx.Response(200, json=ROW)
        if path == "/analyses":
            return httpx.Response(200, json=[ROW])
        if path in ("/polygons", f"/analyses/{AID}/polygons"):
            return httpx.Response(200, json={"type": "FeatureCollection", "features": []})
        if path.startswith("/analyses/"):
            return httpx.Response(404, json={"detail": "analysis not found"})
        return httpx.Response(500)


def make_client(fake: Fake, max_mb: int = 50) -> TestClient:
    settings = Settings(
        "http://inference:8001", "http://geo:8002", max_mb, ["http://localhost:5173"]
    )
    return TestClient(create_app(settings, httpx.MockTransport(fake)))


def upload(client, name="field.jpg", data=b"jpegbytes", bounds="-93.62,41.99,-93.6,42.0"):
    form = {"bounds": bounds, "name": "north"} if bounds is not None else {"name": "north"}
    return client.post("/api/v1/analyses", files={"file": (name, data, "image/jpeg")}, data=form)


def test_health_and_classes():
    with make_client(Fake()) as c:
        assert c.get("/health").json() == {"status": "ok"}
        assert [k["name"] for k in c.get("/api/v1/classes").json()][1] == "nutrient_deficiency"


def test_model_info_proxies_inference():
    with make_client(Fake()) as c:
        assert c.get("/api/v1/model-info").json()["name"] == "unet-resnet34"


def test_upload_happy_path():
    fake = Fake()
    with make_client(fake) as c:
        r = upload(c)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"] == AID
    assert body["model"] == {"name": "unet-resnet34", "version": "3"}
    assert body["stats"] == STATS
    assert "X-Request-ID" in r.headers
    # inference, geo create, geo polygons, geo stats; same request id everywhere
    assert [(q.url.host, q.url.path) for q in fake.calls] == [
        ("inference", "/predict"),
        ("geo", "/analyses"),
        ("geo", f"/analyses/{AID}/polygons"),
        ("geo", "/stats"),
    ]
    assert len({q.headers["X-Request-ID"] for q in fake.calls}) == 1
    created = json.loads(fake.calls[1].content)
    ring = created["image_bounds"]["coordinates"][0]
    assert ring[0] == ring[-1] == [-93.62, 41.99] and ring[2] == [-93.6, 42.0]
    assert created["model_name"] == "unet-resnet34" and created["source_filename"] == "field.jpg"
    assert json.loads(fake.calls[2].content) == PREDICTION["polygons"]


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"name": "field.gif"}, 400),
        ({"bounds": None}, 400),
        ({"bounds": "1,2,3"}, 400),
        ({"bounds": "10,10,0,0"}, 400),
        ({"bounds": "-200,0,0,10"}, 400),
        ({"data": b""}, 400),
    ],
)
def test_upload_validation(kwargs, status):
    fake = Fake()
    with make_client(fake) as c:
        assert upload(c, **kwargs).status_code == status
    assert fake.calls == []  # rejected before calling inference


def test_upload_too_large():
    fake = Fake()
    with make_client(fake, max_mb=1) as c:
        r = upload(c, data=b"x" * (1024 * 1024 + 1))
    assert r.status_code == 413 and fake.calls == []


def test_geotiff_needs_no_bounds():
    fake = Fake()
    with make_client(fake) as c:
        assert upload(c, name="f.tif", bounds=None).status_code == 201
    assert b"bounds" not in fake.calls[0].content


@pytest.mark.parametrize("status", [400, 413])
def test_inference_client_errors_pass_through(status):
    fake = Fake(lambda req: httpx.Response(status, json={"detail": "too big or bad"}))
    with make_client(fake) as c:
        r = upload(c)
    assert r.status_code == status and r.json()["detail"] == "too big or bad"


def test_inference_5xx_is_502():
    with make_client(Fake(lambda req: httpx.Response(500, text="boom"))) as c:
        assert upload(c).status_code == 502


def test_inference_unreachable_is_502():
    def down(req):
        raise httpx.ConnectError("refused")

    with make_client(Fake(down)) as c:
        assert upload(c).status_code == 502


def test_inference_timeout_is_504():
    def slow(req):
        raise httpx.ReadTimeout("slow")

    with make_client(Fake(slow)) as c:
        assert upload(c).status_code == 504


def test_inference_total_deadline_is_504():
    import asyncio

    async def slow(req):
        await asyncio.sleep(0.5)
        return httpx.Response(200, json=PREDICTION)

    settings = Settings("http://inference:8001", "http://geo:8002", 50, [], inference_timeout_s=0.1)
    with TestClient(create_app(settings, httpx.MockTransport(slow))) as c:
        assert upload(c).status_code == 504


def test_read_endpoints():
    with make_client(Fake()) as c:
        one = c.get(f"/api/v1/analyses/{AID}").json()
        assert one["stats"] == STATS and one["model"]["version"] == "3"
        assert c.get("/api/v1/analyses").json()[0]["id"] == AID
        assert c.get(f"/api/v1/analyses/{AID}/polygons").json()["type"] == "FeatureCollection"
        assert c.get("/api/v1/polygons", params={"bbox": "0,0,1,1"}).status_code == 200
        assert c.get("/api/v1/stats", params={"analysis_id": AID}).json() == STATS


def test_unknown_analysis_404():
    with make_client(Fake()) as c:
        r = c.get("/api/v1/analyses/11111111-1111-1111-1111-111111111111")
    assert r.status_code == 404


def test_cors_preflight():
    with make_client(Fake()) as c:
        r = c.options(
            "/api/v1/analyses",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


class FakeUploads:
    """Stands in for S3Uploads: one known upload of 1234 bytes."""

    KEY = "uploads/" + "a" * 32 + ".jpg"

    def new_upload(self, ext, max_bytes):
        return {"upload_key": "uploads/" + "b" * 32 + ext, "url": "https://s3/x", "fields": {}}

    def size(self, key):
        return 1234 if key == self.KEY else None

    def download_url(self, key):
        return f"https://bucket.s3.amazonaws.com/{key}?sig=1"


def aws_client(fake: Fake) -> TestClient:
    settings = Settings("http://inference:8001", "http://geo:8002", 50, [])
    return TestClient(create_app(settings, httpx.MockTransport(fake), uploads=FakeUploads()))


def test_uploads_disabled_locally():
    with make_client(Fake()) as c:
        assert c.post("/api/v1/uploads", json={"filename": "f.jpg"}).status_code == 501


def test_upload_slot_checks_type_and_size():
    with aws_client(Fake()) as c:
        ok = c.post("/api/v1/uploads", json={"filename": "Field.JPG", "size": 10})
        assert ok.status_code == 201 and ok.json()["upload_key"].endswith(".jpg")
        assert c.post("/api/v1/uploads", json={"filename": "f.exe"}).status_code == 400
        big = c.post("/api/v1/uploads", json={"filename": "f.jpg", "size": 51 * 1024 * 1024})
        assert big.status_code == 413


def test_analysis_from_upload_key_sends_url_to_inference():
    fake = Fake()
    with aws_client(fake) as c:
        r = c.post(
            "/api/v1/analyses",
            data={"upload_key": FakeUploads.KEY, "filename": "north.jpg", "bounds": "0,0,1,1"},
        )
    assert r.status_code == 201, r.text
    sent = fake.calls[0].content.decode()
    assert "image_url=https%3A%2F%2Fbucket.s3.amazonaws.com%2Fuploads%2F" in sent
    assert json.loads(fake.calls[1].content)["source_filename"] == "north.jpg"


@pytest.mark.parametrize("key", ["uploads/" + "c" * 32 + ".jpg", "../etc/passwd", "uploads/x.jpg"])
def test_unknown_or_bad_upload_key_400(key):
    fake = Fake()
    with aws_client(fake) as c:
        r = c.post("/api/v1/analyses", data={"upload_key": key, "bounds": "0,0,1,1"})
    assert r.status_code == 400 and fake.calls == []


def test_sigv4_signs_calls(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    fake = Fake()
    settings = Settings(
        "https://inf.lambda-url.ca-central-1.on.aws",
        "https://geo.lambda-url.ca-central-1.on.aws",
        50,
        [],
        aws_region="ca-central-1",
        sign_requests=True,
    )
    with TestClient(create_app(settings, httpx.MockTransport(fake))) as c:
        c.get("/api/v1/model-info")
    auth = fake.calls[0].headers["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/ca-central-1/lambda/aws4_request" in auth
