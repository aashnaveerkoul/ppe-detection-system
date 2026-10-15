
# If True, a person with no helmet/vest detection at all counts as a violation
# (strict). If False, only explicit "no-helmet" / "no-vest" detections do
# (lenient; fewer false positives, more missed violations).
REQUIRE_POSITIVE_DETECTION = True

# Vertical bands of the person box (as fractions of its height, 0 = top)
# where each item's centre should sit. Slightly negative top = allow a helmet
# that sticks out above a tight person box.
HEAD_BAND = (-0.15, 0.45)
TORSO_BAND = (0.10, 0.80)


def _centre_x(box):
    return (box[0] + box[2]) / 2


def _belongs_to(person_box, item_box, band):
    """True if the item's centre lies inside the person box, within the band."""
    px1, py1, px2, py2 = person_box
    height = py2 - py1
    cx = (item_box[0] + item_box[2]) / 2
    cy = (item_box[1] + item_box[3]) / 2
    top = py1 + band[0] * height
    bottom = py1 + band[1] * height
    return px1 <= cx <= px2 and top <= cy <= bottom


def _assign(items, persons, band):
    """Give each item to the single closest person whose band contains it.

    Returns {person_index: [items]}.
    """
    assigned = {i: [] for i in range(len(persons))}
    for item in items:
        best_idx, best_dist = None, float("inf")
        for i, person in enumerate(persons):
            if _belongs_to(person["box"], item["box"], band):
                dist = abs(_centre_x(person["box"]) - _centre_x(item["box"]))
                if dist < best_dist:
                    best_idx, best_dist = i, dist
        if best_idx is not None:
            assigned[best_idx].append(item)
    return assigned


def _best_conf(items):
    return max((it["conf"] for it in items), default=0.0)


def _wearing(positives, negatives):
    """Decide if the item is worn, given positive and negative detections."""
    pos, neg = _best_conf(positives), _best_conf(negatives)
    if pos == 0 and neg == 0:
        return not REQUIRE_POSITIVE_DETECTION  # nothing detected at all
    return pos > neg


def check_compliance(detections):
    """
    detections: Ultralytics Results object (boxes, classes, confidences).

    Returns (worker_status, persons, helmets, vests, no_helmets, no_vests).
    Each worker_status entry has: box, compliant, has_helmet, has_vest, missing.
    """
    boxes = detections.boxes.xyxy.cpu().numpy()
    classes = detections.boxes.cls.cpu().numpy()
    confidences = detections.boxes.conf.cpu().numpy()
    names = detections.names

    persons, helmets, vests, no_helmets, no_vests = [], [], [], [], []

    for box, cls, conf in zip(boxes, classes, confidences):
        class_name = names[int(cls)].lower()
        item = {"box": box, "conf": float(conf)}

        if "person" in class_name:
            persons.append(item)
        elif "no-helmet" in class_name or "no_helmet" in class_name:
            no_helmets.append(item)
        elif "no-vest" in class_name or "no_vest" in class_name:
            no_vests.append(item)
        elif "helmet" in class_name:
            helmets.append(item)
        elif "vest" in class_name:
            vests.append(item)

    helmet_map = _assign(helmets, persons, HEAD_BAND)
    no_helmet_map = _assign(no_helmets, persons, HEAD_BAND)
    vest_map = _assign(vests, persons, TORSO_BAND)
    no_vest_map = _assign(no_vests, persons, TORSO_BAND)

    worker_status = []
    for i, person in enumerate(persons):
        has_helmet = _wearing(helmet_map[i], no_helmet_map[i])
        has_vest = _wearing(vest_map[i], no_vest_map[i])

        missing = []
        if not has_helmet:
            missing.append("helmet")
        if not has_vest:
            missing.append("vest")

        worker_status.append(
            {
                "box": person["box"],
                "compliant": has_helmet and has_vest,
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "missing": missing,
            }
        )

    return worker_status, persons, helmets, vests, no_helmets, no_vests