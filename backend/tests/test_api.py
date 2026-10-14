from pathlib import Path

from app.models.orm import Fingerprint, Observation, ReferenceImage
from tests.conftest import register


def _count(model):
    from app.database import get_sessionmaker

    with get_sessionmaker()() as db:
        return db.query(model).count()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["database"] and body["storage"]
    assert body["embedding"]["available"] is False


def test_register_object_and_fingerprints_persisted(client, img_bytes, settings):
    r = register(client, "Water bottle A", [("a.jpg", img_bytes(1, "same")), ("b.jpg", img_bytes(1, "viewpoint_moderate")),
                                            ("c.jpg", img_bytes(1, "lighting"))], description="dented cap")
    assert r.status_code == 201, r.text
    obj = r.json()["object"]
    assert obj["public_id"].startswith("ODNA-")
    assert obj["image_count"] == 3 and obj["fingerprint_status"] == "complete"
    extractors = {fp["extractor"] for fp in obj["images"][0]["fingerprints"]}
    assert {"orb", "lbp", "color", "gradient"} <= extractors
    fp = obj["images"][0]["fingerprints"][0]
    assert fp["feature_version"] and fp["preprocessing"]["version"] == "prep-v2"
    # Artifacts exist on disk.
    storage = settings.resolved_storage_dir / "objects" / obj["public_id"]
    assert list(storage.rglob("orb.npz")) and list(storage.rglob("global.npz"))
    # Images are served.
    img = client.get(obj["images"][0]["url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"


def test_register_validation_errors(client, img_bytes):
    r = client.post("/api/objects", data={"name": "x", "category": "c"},
                    files=[("files", ("x.txt", b"hello", "text/plain"))])
    assert r.status_code == 422
    r = client.post("/api/objects", data={"name": "", "category": "c"},
                    files=[("files", ("a.jpg", img_bytes(1), "image/jpeg"))])
    assert r.status_code == 422
    assert _count(ReferenceImage) == 0


def test_duplicate_image_rejected_but_duplicate_name_allowed(client, img_bytes):
    a = img_bytes(1)
    assert register(client, "Bottle", [("a.jpg", a)]).status_code == 201
    r = register(client, "Bottle", [("same_file.jpg", a)])
    assert r.status_code == 409
    assert r.json()["detail"]["conflicts"][0]["filename"] == "same_file.jpg"
    r = register(client, "Bottle", [("x.jpg", img_bytes(2)), ("y.jpg", img_bytes(2))])
    assert r.status_code == 409  # same file twice in one request
    # Same name, different physical object -> separate identity.
    r = register(client, "Bottle", [("b.jpg", img_bytes(3))])
    assert r.status_code == 201
    lst = client.get("/api/objects").json()
    assert lst["total"] == 2 and len({o["public_id"] for o in lst["items"]}) == 2


def test_list_search_and_category_filter(client, img_bytes):
    register(client, "Red mug", [("a.jpg", img_bytes(1))], category="Mug")
    register(client, "Blue shoe", [("b.jpg", img_bytes(2))], category="shoe")
    assert client.get("/api/objects", params={"q": "mug"}).json()["total"] == 1
    assert client.get("/api/objects", params={"category": "shoe"}).json()["total"] == 1
    assert set(client.get("/api/objects").json()["categories"]) == {"mug", "shoe"}


def test_identify_match_and_history(client, img_bytes):
    ids = {}
    for seed in (31, 32, 33):
        r = register(client, f"Obj {seed}", [("a.jpg", img_bytes(seed, "same")),
                                              ("b.jpg", img_bytes(seed, "viewpoint_moderate"))])
        ids[seed] = r.json()["object"]["public_id"]
    r = client.post("/api/identify", files={"file": ("q.jpg", img_bytes(32, "lighting"), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "MATCH_FOUND"
    assert body["predicted_object_id"] == ids[32]
    top = body["candidates"][0]
    assert top["object_public_id"] == ids[32]
    assert top["evidence"]["orb"]["inliers"] > 0 and top["matches"]
    assert abs(sum(top["fused"]["contributions"].values()) - top["score"]) < 1e-3
    assert "not calibrated probabilities" in body["note"]
    assert client.get(body["query_image_url"]).status_code == 200
    hist = client.get("/api/history").json()
    assert hist["total"] == 1 and hist["items"][0]["selected_object_public_id"] == ids[32]
    dash = client.get("/api/dashboard").json()
    assert dash["usage"]["query_attempts"] == 1 and dash["usage"]["accepted"] == 1
    assert dash["usage"]["registered_objects"] == 3


def test_identify_unknown_object_not_forced(client, img_bytes):
    register(client, "Known", [("a.jpg", img_bytes(41, "same")), ("b.jpg", img_bytes(41, "viewpoint_moderate"))])
    register(client, "Known 2", [("c.jpg", img_bytes(42, "same"))])
    r = client.post("/api/identify", files={"file": ("q.jpg", img_bytes(77, "same"), "image/jpeg")}).json()
    assert r["status"] != "MATCH_FOUND"
    assert r["predicted_object_id"] is None


def test_identify_threshold_override(client, img_bytes):
    register(client, "Known", [("a.jpg", img_bytes(41, "same"))])
    r = client.post("/api/identify", files={"file": ("q.jpg", img_bytes(41, "lighting"), "image/jpeg")},
                    data={"accept_threshold": "1.0", "uncertain_threshold": "0.99"}).json()
    assert r["status"] != "MATCH_FOUND" and r["thresholds"]["accept"] == 1.0


def test_identify_with_empty_database(client, img_bytes):
    r = client.post("/api/identify", files={"file": ("q.jpg", img_bytes(1), "image/jpeg")})
    assert r.status_code == 200 and r.json()["status"] == "NO_RELIABLE_MATCH"


def test_compare_same_and_different(client, img_bytes):
    same = client.post("/api/compare", files={"file_a": ("a.jpg", img_bytes(5, "same"), "image/jpeg"),
                                              "file_b": ("b.jpg", img_bytes(5, "viewpoint_moderate"), "image/jpeg")})
    assert same.status_code == 200, same.text
    s = same.json()
    assert s["decision"] == "SAME_INSTANCE"
    assert s["evidence"]["orb"]["inliers"] > 10 and any(m["inlier"] for m in s["matches"])
    assert s["image_a"].startswith("data:image/jpeg;base64,")
    diff = client.post("/api/compare", files={"file_a": ("a.jpg", img_bytes(5, "same"), "image/jpeg"),
                                              "file_b": ("b.jpg", img_bytes(9, "same"), "image/jpeg")}).json()
    assert diff["decision"] != "SAME_INSTANCE"
    assert s["score"] > diff["score"]


def test_add_images_and_delete_cascade(client, img_bytes, settings):
    obj = register(client, "Wallet", [("a.jpg", img_bytes(51))]).json()["object"]
    pid = obj["public_id"]
    r = client.post(f"/api/objects/{pid}/images", files=[("files", ("b.jpg", img_bytes(51, "lighting"), "image/jpeg"))])
    assert r.status_code == 201 and r.json()["object"]["image_count"] == 2
    # Remove one image, then refuse to remove the last one.
    img_id = r.json()["object"]["images"][1]["id"]
    assert client.delete(f"/api/objects/{pid}/images/{img_id}").status_code == 204
    last = client.get(f"/api/objects/{pid}").json()["images"][0]["id"]
    assert client.delete(f"/api/objects/{pid}/images/{last}").status_code == 409

    obs = client.post(f"/api/objects/{pid}/observations",
                      files={"file": ("o.jpg", img_bytes(51, "viewpoint_moderate"), "image/jpeg")}).json()
    assert obs["accepted"] is True
    assert _count(Fingerprint) > 0 and _count(Observation) == 1
    assert client.delete(f"/api/objects/{pid}").status_code == 204
    assert client.get(f"/api/objects/{pid}").status_code == 404
    assert _count(ReferenceImage) == 0 and _count(Fingerprint) == 0 and _count(Observation) == 0
    assert not (Path(settings.resolved_storage_dir) / "objects" / pid).exists()
    assert client.delete(f"/api/objects/{pid}").status_code == 404


def test_observation_history_and_rejection(client, img_bytes):
    a = register(client, "Mug A", [("a.jpg", img_bytes(61)), ("a2.jpg", img_bytes(61, "viewpoint_moderate"))]).json()["object"]
    b = register(client, "Mug B", [("b.jpg", img_bytes(62))]).json()["object"]
    ok = client.post(f"/api/objects/{a['public_id']}/observations",
                     files={"file": ("o.jpg", img_bytes(61, "lighting"), "image/jpeg")}, data={"note": "after trip"}).json()
    assert ok["accepted"] and ok["observation"]["surface_changes"]["keypoints"] > 0
    # An image of object A submitted as an observation of B must not be accepted.
    bad = client.post(f"/api/objects/{b['public_id']}/observations",
                      files={"file": ("o2.jpg", img_bytes(61, "background"), "image/jpeg")}).json()
    assert bad["accepted"] is False
    h = client.get(f"/api/objects/{a['public_id']}/history").json()
    assert len(h["observations"]) == 1 and h["observations"][0]["note"] == "after trip"
    assert "not by themselves evidence of damage" in h["note"]
    assert client.get(h["observations"][0]["image_url"]).status_code == 200


def test_decision_settings_update_and_reset(client):
    r = client.put("/api/settings/decision", json={"accept_threshold": 0.7, "uncertain_threshold": 0.4,
                                                   "margin_threshold": 0.1})
    assert r.status_code == 200 and r.json()["accept_threshold"] == 0.7
    assert client.get("/api/settings/fusion").json()["accept_threshold"] == 0.7
    bad = client.put("/api/settings/decision", json={"accept_threshold": 0.3, "uncertain_threshold": 0.4,
                                                     "margin_threshold": 0.1})
    assert bad.status_code == 422
    assert client.post("/api/settings/fusion/reset").json()["source"] == "default-heuristic"


def test_openapi_lists_required_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/api/objects", "/api/objects/{object_id}", "/api/objects/{object_id}/images", "/api/identify",
              "/api/compare", "/api/history", "/api/dashboard", "/api/experiments/run", "/api/experiments",
              "/api/health"):
        assert p in paths
