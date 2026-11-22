"""End-to-end tests against a real PostGIS database (see conftest.py)."""

import uuid

import pytest

BOUNDS = {
    "type": "Polygon",
    "coordinates": [
        [[-93.62, 41.99], [-93.61, 41.99], [-93.61, 42.0], [-93.62, 42.0], [-93.62, 41.99]]
    ],
}


def square(lon: float, lat: float, d: float = 0.001) -> list:
    return [[[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]]


def feature(class_id: int, name: str, coords: list, multi: bool = False) -> dict:
    geometry = (
        {"type": "MultiPolygon", "coordinates": [coords]}
        if multi
        else {"type": "Polygon", "coordinates": coords}
    )
    return {
        "type": "Feature",
        "geometry": geometry,
        "properties": {
            "class_id": class_id,
            "class_name": name,
            "confidence": 0.8,
            "area_m2": 9000.0,
        },
    }


@pytest.fixture
def analysis(client) -> dict:
    r = client.post(
        "/analyses",
        json={
            "name": "north-field",
            "source_filename": "field.jpg",
            "model_name": "unet-resnet34",
            "model_version": "3",
            "image_bounds": BOUNDS,
        },
    )
    assert r.status_code == 201, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_get_analysis(client, analysis):
    assert analysis["status"] == "completed"
    assert analysis["image_bounds"]["type"] == "Polygon"
    uuid.UUID(analysis["id"])
    got = client.get(f"/analyses/{analysis['id']}").json()
    assert got == analysis
    ids = [a["id"] for a in client.get("/analyses").json()]
    assert ids[0] == analysis["id"]  # newest first


def test_unknown_analysis_404(client):
    assert client.get(f"/analyses/{uuid.uuid4()}").status_code == 404
    r = client.post(
        f"/analyses/{uuid.uuid4()}/polygons", json={"type": "FeatureCollection", "features": []}
    )
    assert r.status_code == 404


def test_polygons_and_stats(client, analysis):
    aid = analysis["id"]
    fc = {
        "type": "FeatureCollection",
        "features": [
            feature(1, "nutrient_deficiency", square(-93.62, 41.99)),
            feature(1, "nutrient_deficiency", square(-93.615, 41.995), multi=True),
            feature(3, "weed_cluster", square(-93.612, 41.991)),
        ],
    }
    r = client.post(f"/analyses/{aid}/polygons", json=fc)
    assert r.status_code == 201 and r.json() == {"inserted": 3}

    got = client.get(f"/analyses/{aid}/polygons").json()
    assert len(got["features"]) == 3
    first = got["features"][0]
    assert first["geometry"]["type"] == "MultiPolygon"  # stored as MultiPolygon
    assert first["properties"]["analysis_id"] == aid
    assert isinstance(first["properties"]["id"], int)

    stats = client.get("/stats", params={"analysis_id": aid}).json()
    assert [(s["class_id"], s["polygon_count"]) for s in stats] == [(1, 2), (3, 1)]
    # 0.001 deg square near 42N: about 111 m x 83 m = 0.92 ha (geodesic, not degrees)
    one = stats[1]["area_ha"]
    assert 0.85 < one < 1.0

    near = client.get(
        "/polygons", params={"bbox": "-93.6205,41.9895,-93.6195,41.9905", "analysis_id": aid}
    ).json()
    assert len(near["features"]) == 1
    weeds = client.get("/polygons", params={"class_id": 3, "analysis_id": aid}).json()
    assert [f["properties"]["class_name"] for f in weeds["features"]] == ["weed_cluster"]


def test_empty_feature_collection(client, analysis):
    r = client.post(
        f"/analyses/{analysis['id']}/polygons", json={"type": "FeatureCollection", "features": []}
    )
    assert r.json() == {"inserted": 0}
    assert client.get("/stats", params={"analysis_id": analysis["id"]}).json() == []


def test_bad_bbox_400(client):
    assert client.get("/polygons", params={"bbox": "1,2,3"}).status_code == 400
    assert client.get("/stats", params={"bbox": "5,5,1,1"}).status_code == 400
