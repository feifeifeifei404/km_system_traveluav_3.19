from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("/mnt/data/TravelUAV")

METHODS = {
    "dino_semantic": ROOT / "result/eval_closeloop/eval_town01",
    "dsp": ROOT / "result_dsp_152/eval_closeloop/eval_town01",
    "dsp_depth": ROOT / "result_dsp_depth_152/eval_closeloop/eval_town01",
    "ICnet": ROOT / "result_icnet_152/eval_closeloop/eval_town01",
    "dino_uncertainty": ROOT / "result_dino_uncertainty_152/eval_closeloop/eval_town01",
}

TASKS = [
    "c30260ba-1b02-41a8-9b62-da1c459a33ed",
    "2258de49-6868-4b36-a37f-45f0948789b8",
    "3db5d587-dbb8-43ed-91a0-7b6affee943c",
    "0ec8f55e-5f51-4e30-8ec3-e2b12173b354",
]

COLORS = {
    "dino_semantic": "#d94f62",
    "dsp": "#087f95",
    "dsp_depth": "#f2b447",
    "ICnet": "#356b9a",
    "dino_uncertainty": "#9b315f",
}


def find_task_dir(base: Path, task_id: str):
    candidates = [
        base / task_id,
        base / f"success_{task_id}",
        base / f"oracle_{task_id}",
    ]
    for p in candidates:
        if p.exists():
            return p

    # 兜底：只要目录名包含 task_id 就接受
    hits = [p for p in base.iterdir() if p.is_dir() and task_id in p.name]
    if hits:
        return hits[0]

    return None


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def extract_curve(complexity_json):
    """
    尽量兼容不同 JSON 格式：
    1. list[float]
    2. list[dict]
    3. dict with scores/list/items
    """
    data = complexity_json

    if isinstance(data, dict):
        for key in [
            "complexity_scores",
            "scores",
            "complexities",
            "complexity",
            "data",
            "frames",
            "records",
        ]:
            if key in data:
                data = data[key]
                break

    xs, ys = [], []

    if isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, (int, float)):
                xs.append(i)
                ys.append(float(item))
            elif isinstance(item, dict):
                x = (
                    item.get("logged_step")
                    or item.get("frame")
                    or item.get("step")
                    or item.get("idx")
                    or item.get("index")
                    or i
                )
                y = (
                    item.get("complexity")
                    or item.get("complexity_score")
                    or item.get("score")
                    or item.get("value")
                )
                if y is not None:
                    xs.append(float(x))
                    ys.append(float(y))

    elif isinstance(data, dict):
        # dict 可能是 {"0": 0.1, "5": 0.2, ...}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                xs.append(float(k) if str(k).replace(".", "", 1).isdigit() else len(xs))
                ys.append(float(v))
            elif isinstance(v, dict):
                y = (
                    v.get("complexity")
                    or v.get("complexity_score")
                    or v.get("score")
                    or v.get("value")
                )
                if y is not None:
                    xs.append(float(k) if str(k).replace(".", "", 1).isdigit() else len(xs))
                    ys.append(float(y))

    return np.array(xs), np.array(ys)


def find_metric(d, names):
    if not isinstance(d, dict):
        return None

    for name in names:
        if name in d and isinstance(d[name], (int, float)):
            return d[name]

    # 递归找
    for v in d.values():
        if isinstance(v, dict):
            out = find_metric(v, names)
            if out is not None:
                return out

    return None


def load_task_stats(task_dir: Path):
    stats = {}

    for filename in ["evaluation_results.json", "ori_info.json"]:
        p = task_dir / filename
        if not p.exists():
            continue

        d = load_json(p)

        if "gap" not in stats:
            stats["gap"] = find_metric(d, ["gap", "final_gap", "distance_gap"])

        if "steps" not in stats:
            stats["steps"] = find_metric(d, ["steps", "step", "num_steps", "logged_steps"])

        if "llm" not in stats:
            stats["llm"] = find_metric(d, ["LLM", "llm", "llm_count", "num_llm", "llm_calls"])

    return stats


def plot_one_task(task_id: str, out_dir: Path):
    curves = {}
    all_stats = []

    for method, base in METHODS.items():
        task_dir = find_task_dir(base, task_id)
        if task_dir is None:
            print(f"[MISS] {method}: {task_id}")
            continue

        complexity_path = task_dir / "complexity_scores.json"
        if not complexity_path.exists():
            print(f"[MISS] {method}: {complexity_path}")
            continue

        complexity_json = load_json(complexity_path)
        x, y = extract_curve(complexity_json)

        if len(y) == 0:
            print(f"[EMPTY] {method}: {complexity_path}")
            continue

        curves[method] = (x, y, task_dir)

        stats = load_task_stats(task_dir)
        if len(y) > 0:
            stats["avg_complexity"] = float(np.mean(y))
        all_stats.append(stats)

    if not curves:
        print(f"[SKIP] no curves for {task_id}")
        return

    avg_complexity = np.mean([
        s["avg_complexity"] for s in all_stats if s.get("avg_complexity") is not None
    ])

    gaps = [s["gap"] for s in all_stats if s.get("gap") is not None]
    steps = [s["steps"] for s in all_stats if s.get("steps") is not None]
    llms = [s["llm"] for s in all_stats if s.get("llm") is not None]

    gap_text = f"{np.mean(gaps):.3f}" if gaps else "NA"
    step_text = f"{np.mean(steps):.2f}" if steps else "NA"
    llm_text = f"{np.mean(llms):.2f}" if llms else "NA"

    plt.figure(figsize=(14, 7))

    for method, (x, y, task_dir) in curves.items():
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2.4,
            markersize=4.5,
            label=method,
            color=COLORS.get(method),
        )

    short_id = task_id[:8] + "..." + task_id[-6:]
    plt.title(f"Task {short_id}", fontsize=22, fontweight="bold", loc="left")

    subtitle = (
        f"avg complexity={avg_complexity:.3f}; "
        f"gap={gap_text}; "
        f"avg steps={step_text}; "
        f"avg LLM={llm_text}"
    )
    plt.text(
        0.0,
        1.03,
        subtitle,
        transform=plt.gca().transAxes,
        fontsize=13,
        color="#555555",
    )

    plt.xlabel("frame / logged step", fontsize=14)
    plt.ylabel("complexity score", fontsize=14)
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.10),
        ncol=5,
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()

    out_path = out_dir / f"complexity_task_{task_id}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[SAVE] {out_path}")


out_dir = ROOT / "plots_complexity_4tasks"
out_dir.mkdir(parents=True, exist_ok=True)

for task_id in TASKS:
    plot_one_task(task_id, out_dir)

print("Done:", out_dir)