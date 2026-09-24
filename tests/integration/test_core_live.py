"""cinematlas.core against real Atlas + Voyage: joint vectors, filters, moments, image queries, upserts."""

import pytest
from PIL import Image as PILImage

from cinematlas.core import Atlas, Image, Text

pytestmark = pytest.mark.integration

RECORDS = [
    {"sku": "r1", "title": "Trail running shoe", "colour": "red", "category": "shoes",
     "description": "Grippy lugs for mud. A breathable mesh upper keeps feet cool."},
    {"sku": "b1", "title": "Canvas tote bag", "colour": "blue", "category": "bags",
     "description": "Holds a laptop and groceries. Machine washable cotton."},
    {"sku": "g1", "title": "Yoga mat", "colour": "green", "category": "fitness",
     "description": "Six millimetres of cushioning. Non-slip on both sides."},
]


@pytest.fixture
def products(backend, run_id):
    eng, _ = backend
    atlas = Atlas(mongo_client=eng.collection.database.client, voyage_client=eng.vo)
    coll = atlas.collection("cinematlas_ci.core_products", embed=Text("title") + Image("swatch"),
                            moment="description", key="sku", filters=["category", "run"])
    coll.setup(timeout_s=900)
    yield coll, run_id
    coll.delete({"run": run_id})


def test_joint_vectors_search_filter_and_find_the_sentence(products):
    coll, run = products
    records = [{**r, "sku": f"{run}-{r['sku']}", "run": run, "swatch": PILImage.new("RGB", (64, 64), r["colour"])}
               for r in RECORDS]
    assert coll.add(records).added == 3
    coll.wait_until_searchable(timeout_s=180)

    hits = coll.search("shoes for running in mud", k=3, where={"run": run})
    assert hits.top["sku"] == f"{run}-r1"
    assert "mud" in hits.top.text.lower(), "the reranker picks the sentence that answers"
    assert "swatch" not in hits.top and "embedding" not in hits.top

    only_bags = coll.search("something to carry a laptop", k=3, where={"run": run, "category": "bags"})
    assert [h["category"] for h in only_bags] == ["bags"]

    by_picture = coll.search(PILImage.new("RGB", (64, 64), "green"), k=3, where={"run": run})
    assert by_picture.top["sku"] == f"{run}-g1", "an image query finds the record with that swatch"

    coll.add([{**records[0], "title": "Trail running shoe v2"}])  # same key: replaced, not duplicated
    assert coll.count({"run": run}) == 3
