"""Joint-vector search beyond video: NASA's photo library, title + photo in one vector.

    uv run python examples/photos.py "astronaut fixing a telescope in space"
    uv run python examples/photos.py "a rover's tracks on red sand" --center JPL
    uv run python examples/photos.py --like path/to/photo.jpg

The first run downloads ~200 public-domain NASA photos and indexes them (a minute or two).
Each record's title and photo are embedded *together*, so one vector answers questions about
either; the reranker then picks the sentence of the description that fits best.
"""

import argparse
import json
import urllib.parse
import urllib.request

from _env import engine  # noqa: F401  (loads .env and checks keys)

from cinematlas.core import Atlas, Image, Text

TOPICS = ["hubble servicing", "apollo moon", "mars rover", "space station", "shuttle launch",
          "spacewalk", "earth from orbit", "saturn cassini"]

p = argparse.ArgumentParser()
p.add_argument("question", nargs="*")
p.add_argument("--center", help="filter by NASA center, e.g. JPL, JSC, GSFC, KSC")
p.add_argument("--like", help="search by an image instead of text")
p.add_argument("-k", type=int, default=3)
args = p.parse_args()


def nasa_photos(per_topic: int = 25):
    """Public-domain records from images.nasa.gov: title, description, center and a thumbnail URL."""
    for topic in TOPICS:
        url = "https://images-api.nasa.gov/search?" + urllib.parse.urlencode(
            {"q": topic, "media_type": "image", "page_size": per_topic})
        with urllib.request.urlopen(url, timeout=30) as r:
            items = json.load(r)["collection"]["items"]
        for item in items:
            data = item["data"][0]
            preview = next((link["href"] for link in item.get("links", []) if link.get("rel") == "preview"), None)
            if preview:  # a ~thumb.jpg, a few dozen KB
                yield {"nasa_id": data["nasa_id"], "title": data.get("title", ""),
                       "description": data.get("description", ""), "center": data.get("center"),
                       "date": data.get("date_created", "")[:10], "image": preview, "topic": topic}


with Atlas() as atlas:
    photos = atlas.collection("cinematlas_examples.photos", embed=Text("title") + Image("image"),
                              moment="description", key="nasa_id", filters=["center"],
                              display=("title",))
    photos.setup()
    if photos.count() == 0:
        print("Indexing NASA photos (first run only)…")
        result = photos.add(nasa_photos(), progress=lambda seen, added: print(
            f"\r  {seen} seen, {added} added", end="", flush=True))
        print(f"\n{result}")
        photos.wait_until_searchable()

    query = Image(args.like) if args.like else " ".join(args.question) or "astronaut fixing a telescope in space"
    hits = photos.search(query, k=args.k, where={"center": args.center} if args.center else None)

print(f"\n{'Image: ' + args.like if args.like else repr(query)}\n")
for i, hit in enumerate(hits, 1):
    print(f"{i}. {hit.title}  [{hit.center}, {hit.date}]  score {hit.score:.3f}")
    if hit.text:
        print(f"   “{hit.text[:140]}”")
    print(f"   {hit.image}\n")
