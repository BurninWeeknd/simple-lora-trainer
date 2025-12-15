import math

def parse_int(form, key, issues, min_value=None, max_value=None):
    raw = form.get(key, "").strip()

    if raw == "":
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} cannot be empty. Value has been reverted."
        })
        return None

    try:
        value = int(raw)
    except Exception:
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} must be an integer. Value has been reverted."
        })
        return None

    if min_value is not None and value < min_value:
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} must be ≥ {min_value}. Value has been reverted."
        })
        return None

    if max_value is not None and value > max_value:
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} is unreasonably large. Value has been reverted."
        })
        return None

    return value

def parse_float(form, key, issues, min_value=None, max_value=None):
    raw = form.get(key, "").strip()

    if raw == "":
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} cannot be empty. Value has been reverted."
        })
        return None

    try:
        value = float(raw)
    except Exception:
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} must be a number. Value has been reverted."
        })
        return None

    if not math.isfinite(value):
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} must be a finite number. Value has been reverted."
        })
        return None

    if min_value is not None and value < min_value:
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} must be ≥ {min_value}. Value has been reverted."
        })
        return None

    if max_value is not None and value > max_value:
        issues.append({
            "field": key,
            "level": "fatal",
            "message": f"{key.replace('_', ' ').title()} is unreasonably large. Value has been reverted."
        })
        return None

    return value

