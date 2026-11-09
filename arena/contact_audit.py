"""Conservative contact-quality check of trial telemetry.

The simulator's raw bump channel excludes jaws/pads because they normally touch
objects. This check includes jaw/pad contact with furniture and other robot
links. It does not reinterpret object gripping as a furniture bump.
"""


def peak(row):
    highest = float(row.get("max_bump", 0))
    for pair, force in row.get("diagnostics", {}).get("contacts", []):
        names = pair.split(" / ")
        if any(n.startswith(("obj_", "rob_tyre", "rob_cast", "per_")) for n in names):
            continue
        highest = max(highest, force)
    return highest


def clear(row):
    contacts = row.get("diagnostics", {}).get("contacts", [])
    # Only the 12 highest contact pairs are retained. If the last is still over
    # the limit, an omitted furniture contact could be hidden by object grips.
    complete = len(contacts) < 12 or contacts[-1][1] <= 40
    return bool(complete and peak(row) <= 40)


def clean(row):
    return bool(row.get("ok") and clear(row))
