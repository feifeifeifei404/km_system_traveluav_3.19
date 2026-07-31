from pathlib import Path

BASE = Path("/mnt/data/TravelUAV/result_seen/result_town01_54.61%")

UUIDS_TEXT = """
08302fc9-8117-434e-a77f-46cabecf6571
0a7a07a1-0757-4528-9463-ca6c0eb2ec92
0d4eb894-ed4e-4a11-8bff-24ea3b1030bb
0ec8f55e-5f51-4e30-8ec3-e2b12173b354
106c9330-f41e-4c17-95b9-93caf8f17ee3
136376ab-6f60-4c80-ac80-1ad2ceb9c062
139237bc-00e6-4c86-a766-29e7ebd7c333
13bb9524-b181-4ff5-83de-49b5405c17d0
13e34434-fedc-4653-a38b-6011e10ee205
14264108-503a-44ef-bf4c-6396c5bc098d
1592c21a-ad06-454f-b694-ecad9621df9d
1d6bb68f-8281-419a-90a6-8c3333b668f1
20375e27-9b65-45a3-bfd8-4a79b1069cf5
2258de49-6868-4b36-a37f-45f0948789b8
23b3c060-5ac9-4c49-9470-c15ff3e3be9a
2adad6e8-d896-40ef-931d-30722efd9522
2c84256d-c3fa-48e9-9370-50731dbef63a
33ebeb3e-fc04-4f26-8a4f-d6dbdb07bfd0
35a7cd61-bcdf-4884-b509-614330cced22
36d9f9a4-8190-454e-ae71-2d7db7409636
39b3a13a-c82c-4ab2-a25a-40ff66a0ad36
3b53ba7c-63b4-44be-bae3-40d263a4d256
3db5d587-dbb8-43ed-91a0-7b6affee943c
411cc211-236e-4b76-8c10-0922c7ac8c9f
4c99d620-1c1c-4cc9-8bec-b4853318fff8
4da06af0-95b7-4fd3-8c21-c31315857138
54908ca6-ed10-43fa-86d7-d0971f211342
66a02730-e8e2-476f-8a41-74500ae001fc
69ac92f1-3eff-4972-a187-ebeb2adff8fc
6a1b7f31-aedc-4dde-a98a-8027755d3283
7eb4f1aa-bd11-4edc-99d8-ba7893cff09a
86dfd9d2-b91f-4946-b142-829ffb7e0e0a
8a6e8c92-3763-417d-9ab9-359370ba5db6
8c772c2a-b718-416f-962e-49795fcc897a
94529e15-3455-4a03-80be-84b70d7ea0d9
9750822c-60c9-4695-8a95-86b8b93cb671
99617088-214f-43cb-8d0d-0c5551b00059
9de262ec-48d3-475c-b607-a7f92d9bf4eb
a43cfb70-9ca7-4584-8cfd-43b109084aa2
a915fc5c-027b-4e5b-bbe9-a6c51b2ac2a3
aa137a79-c5b8-41b3-b4f8-2532767cdd85
ac96c2d1-4fa6-4b4b-b41a-0e4b97f07303
ae602b76-6d75-466a-9fd1-987abee73d0d
b0713c1f-67ec-40b6-b6df-3ee6289df8b0
b379b22a-7004-4216-815f-a7f6189e97d3
babd875c-2c58-46ce-b4c1-c26051377896
bb038237-392e-4f51-9948-ea903dadc99b
c30260ba-1b02-41a8-9b62-da1c459a33ed
c659cd7a-e4c0-4fc6-9183-19ca9ce2d657
c783552d-5307-4087-81b6-3cb87b93f13c
c96dc866-7853-43f1-9211-55153578baef
c99dfc69-f19d-4d4e-bc2c-85d715791b72
cc14c81a-18ed-4519-b96d-54f8a0db5d32
cd5ea717-a5ca-4c89-b814-7e637911515d
cd6bf4a8-befa-45a1-a77c-42f3cc7aac1b
d196267b-ff8b-4c24-a9b2-14e3fde04fe4
d3667057-4eae-4fb4-82c7-d99cfa93ec9d
d4815c27-84c3-4134-acaa-4fe6fa63d375
d6a7cc6d-24a4-4731-bc1d-488f6575c7a2
df24d535-4158-4e9f-9226-4535e928336a
e882e620-a645-44e9-8105-f66dfe4e128b
e8f41596-74cd-4980-8ff2-6569bf004d09
eb02b72e-593d-4df3-a116-1ea6e7c5f133
ec219aa8-c0d5-4f70-9f10-3e861b5c3e74
ee3d73d5-1061-497e-8a45-9b340d8e092d
ee6ef655-c6e7-44b1-a168-b9e456236834
f207934b-365f-4b1f-8443-5a5e569d62a6
f9bdc59d-6905-43cf-b4c0-dd26512fe7a9
fbf51d6f-19ed-4fc1-81fb-5af0690fe723
fe8e161f-099a-4363-9697-e15609d79ff7
"""

def strip_prefix(name: str) -> str:
    for prefix in ["success_", "oracle_", "fail_"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def find_eval_dir(base: Path) -> Path:
    candidates = [
        base,
        base / "eval_closeloop" / "eval_town01",
        base / "eval_closeloop" / "eval_Town01",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot find valid result dir under: {base}")

uuids = [x.strip() for x in UUIDS_TEXT.splitlines() if x.strip()]
seen = set()
dups = []
for u in uuids:
    if u in seen:
        dups.append(u)
    seen.add(u)

eval_dir = find_eval_dir(BASE)

index = {}
duplicate_found = {}

for p in eval_dir.iterdir():
    if not p.is_dir():
        continue
    uuid = strip_prefix(p.name)
    success = p.name.startswith("success_")
    if uuid in index:
        duplicate_found.setdefault(uuid, []).append(p.name)
    else:
        index[uuid] = {
            "dirname": p.name,
            "success": success,
        }

total_input = len(uuids)
found = []
missing = []

for u in uuids:
    if u in index:
        found.append(u)
    else:
        missing.append(u)

success_ids = [u for u in found if index[u]["success"]]
fail_ids = [u for u in found if not index[u]["success"]]

found_total = len(found)
succ = len(success_ids)
fail = len(fail_ids)
rate_found = succ / found_total * 100 if found_total else 0.0
rate_input = succ / total_input * 100 if total_input else 0.0

print("=== Selected UUID baseline success statistics ===")
print(f"eval_dir        : {eval_dir}")
print(f"input UUIDs     : {total_input}")
print(f"unique UUIDs    : {len(set(uuids))}")
print(f"found           : {found_total}")
print(f"missing         : {len(missing)}")
print(f"success         : {succ}")
print(f"non-success     : {fail}")
print(f"success rate(found only) : {succ}/{found_total} = {rate_found:.2f}%")
print(f"success rate(all input)  : {succ}/{total_input} = {rate_input:.2f}%")

if dups:
    print("\n[WARN] duplicate UUIDs in input:")
    for u in dups:
        print(u)

if duplicate_found:
    print("\n[WARN] duplicate matched result dirs:")
    for u, names in duplicate_found.items():
        print(u, names)

out_dir = Path("/mnt/data/TravelUAV")
(out_dir / "selected_seen_success_ids.txt").write_text("\n".join(success_ids) + "\n")
(out_dir / "selected_seen_fail_ids.txt").write_text("\n".join(fail_ids) + "\n")
(out_dir / "selected_seen_missing_ids.txt").write_text("\n".join(missing) + "\n")

print("\n[saved]")
print(out_dir / "selected_seen_success_ids.txt")
print(out_dir / "selected_seen_fail_ids.txt")
print(out_dir / "selected_seen_missing_ids.txt")

if missing:
    print("\n=== Missing UUIDs ===")
    for u in missing:
        print(u)

print("\n=== Non-success UUIDs ===")
for u in fail_ids:
    print(f"{u}  ->  {index[u]['dirname']}")
