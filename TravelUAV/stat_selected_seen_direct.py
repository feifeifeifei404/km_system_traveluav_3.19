from pathlib import Path

EVAL_DIR = Path("/mnt/data/TravelUAV/result_seen/result_town01_54.61%/eval_closeloop/eval_town01")
UUID_FILE = Path("/mnt/data/TravelUAV/selected_seen_missing_ids.txt")

PREFIXES = ["success_", "oracle_", "fail_"]

def strip_prefix(name: str) -> str:
    for prefix in PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

uuids = [x.strip() for x in UUID_FILE.read_text().splitlines() if x.strip()]
target = set(uuids)

index = {}

for p in EVAL_DIR.iterdir():
    if not p.is_dir():
        continue

    uuid = strip_prefix(p.name)

    if uuid in target:
        index[uuid] = {
            "dirname": p.name,
            "success": p.name.startswith("success_"),
            "path": str(p),
        }

found = [u for u in uuids if u in index]
missing = [u for u in uuids if u not in index]
success_ids = [u for u in found if index[u]["success"]]
nonsuccess_ids = [u for u in found if not index[u]["success"]]

total = len(uuids)
found_total = len(found)
succ = len(success_ids)
fail = len(nonsuccess_ids)

rate_found = succ / found_total * 100 if found_total else 0
rate_all = succ / total * 100 if total else 0

print("=== selected baseline statistics ===")
print(f"eval_dir        : {EVAL_DIR}")
print(f"input UUIDs     : {total}")
print(f"found           : {found_total}")
print(f"missing         : {len(missing)}")
print(f"success         : {succ}")
print(f"non-success     : {fail}")
print(f"success rate(found only): {succ}/{found_total} = {rate_found:.2f}%")
print(f"success rate(all input) : {succ}/{total} = {rate_all:.2f}%")

print("\n=== success UUIDs ===")
for u in success_ids:
    print(f"{u} -> {index[u]['dirname']}")

print("\n=== non-success UUIDs ===")
for u in nonsuccess_ids:
    print(f"{u} -> {index[u]['dirname']}")

if missing:
    print("\n=== missing UUIDs ===")
    for u in missing:
        print(u)

OUT = Path("/mnt/data/TravelUAV")
(OUT / "selected_seen_success_ids.txt").write_text("\n".join(success_ids) + "\n")
(OUT / "selected_seen_nonsuccess_ids.txt").write_text("\n".join(nonsuccess_ids) + "\n")
(OUT / "selected_seen_missing_ids_v2.txt").write_text("\n".join(missing) + "\n")

print("\n[saved]")
print(OUT / "selected_seen_success_ids.txt")
print(OUT / "selected_seen_nonsuccess_ids.txt")
print(OUT / "selected_seen_missing_ids_v2.txt")
