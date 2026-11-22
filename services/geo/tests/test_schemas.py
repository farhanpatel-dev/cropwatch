import pytest
from pydantic import ValidationError

from app.schemas import FeatureCollection, parse_bbox


def test_parse_bbox_ok():
    assert parse_bbox("-93.7,41.9,-93.5,42.1") == (-93.7, 41.9, -93.5, 42.1)


@pytest.mark.parametrize(
    "text", ["1,2,3", "a,b,c,d", "1,1,0,2", "-200,0,10,10", "0,-95,10,10", "0,0,10,10,5"]
)
def test_parse_bbox_rejects(text):
    with pytest.raises(ValueError):
        parse_bbox(text)


def test_feature_collection_rejects_point():
    with pytest.raises(ValidationError):
        FeatureCollection.model_validate(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [0, 0]},
                        "properties": {"class_id": 1, "class_name": "drydown"},
                    }
                ],
            }
        )


def test_confidence_must_be_0_to_1():
    with pytest.raises(ValidationError):
        FeatureCollection.model_validate(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": []},
                        "properties": {"class_id": 1, "class_name": "x", "confidence": 1.5},
                    }
                ],
            }
        )
