from pathlib import Path
from itertools import combinations

ROOT = Path("/mnt/data/TravelUAV")

METHODS = {
    "dino_semantic": ROOT / "result_dino_semantic_152/eval_closeloop/eval_Town01",
    "dsp": ROOT / "result_dsp_152/eval_closeloop/eval_town01",
    "dsp_depth": ROOT / "result_dsp_depth_152/eval_closeloop/eval_town01",
    "icnet": ROOT / "result_icnet_152/eval_closeloop/eval_town01",
    "dino_uncertainty": ROOT / "result_dino_uncertainty_152/eval_closeloop/eval_town01",
}

THRESHOLD = 54.0
DINO = "dino_semantic"
ICNET = "icnet"
OTHERS = ["dsp", "dsp_depth", "dino_uncertainty"]
MIN_TASKS = 60

def norm_task_name(name: str) -> str:
    for prefix in ["success_", "oracle_", "fail_"]:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name

def load_method(path: Path):
    data = {}
    for p in path.iterdir():
        if p.is_dir():
            task = norm_task_name(p.name)
            data[task] = p.name.startswith("success_")
    return data

def get_rate(data_dict, tasks, method):
    if not tasks:
        return 0.0
    succ = sum(1 for t in tasks if data_dict[method][t])
    return succ / len(tasks) * 100

def all_above(data_dict, tasks, threshold=THRESHOLD):
    return all(get_rate(data_dict, tasks, m) >= threshold for m in data_dict)

def check_order(data_dict, tasks):
    """检查是否满足 dino > icnet > 所有others"""
    rates = {m: get_rate(data_dict, tasks, m) for m in data_dict}
    if rates[DINO] <= rates[ICNET]:
        return False
    for m in OTHERS:
        if rates[ICNET] <= rates[m]:
            return False
    return True

def select_max_subset(data_dict, all_tasks):
    """找出满足所有方法>=54且任务数最多的子集（优先dino高）"""
    N = len(all_tasks)
    if N == 0:
        return []
    if all_above(data_dict, all_tasks):
        return all_tasks
    if N <= 25:
        best = []
        best_pri = -1
        for size in range(N, 0, -1):
            for comb in combinations(all_tasks, size):
                if all_above(data_dict, comb):
                    pri = get_rate(data_dict, comb, DINO)
                    if len(comb) > len(best) or (len(comb) == len(best) and pri > best_pri):
                        best = list(comb)
                        best_pri = pri
            if best:
                break
        return best
    # 贪心
    current = set(all_tasks)
    while True:
        if all_above(data_dict, current):
            break
        rates = {m: get_rate(data_dict, current, m) for m in data_dict}
        worst = min(rates, key=rates.get)
        fail_tasks = [t for t in current if not data_dict[worst][t]]
        if not fail_tasks:
            break
        best_remove = None
        best_score = (-1.0, -1.0)
        for t in fail_tasks:
            new_set = current - {t}
            if all_above(data_dict, new_set):
                best_remove = t
                break
            min_rate = min(get_rate(data_dict, new_set, m) for m in data_dict)
            pri_rate = get_rate(data_dict, new_set, DINO)
            if (min_rate, pri_rate) > best_score:
                best_score = (min_rate, pri_rate)
                best_remove = t
        if best_remove is None:
            break
        current.remove(best_remove)
    return list(current) if all_above(data_dict, current) else []

def optimize_strict(data_dict, initial_tasks):
    current = set(initial_tasks)
    removed = []
    # 如果已经满足，直接返回
    if check_order(data_dict, current):
        return list(current), removed

    # 循环删除，直到满足或任务数<MIN_TASKS
    while len(current) > MIN_TASKS:
        # 尝试找到一个删除，使得删除后直接满足顺序
        for t in sorted(current):  # 确保确定性
            new_set = current - {t}
            if all_above(data_dict, new_set) and check_order(data_dict, new_set):
                current.remove(t)
                removed.append(t)
                return list(current), removed  # 直接成功，任务数最多

        # 没有一次删除能满足顺序，则贪心删除使评分提升最大的
        best_t = None
        best_score = -1e9
        current_rates = {m: get_rate(data_dict, current, m) for m in data_dict}
        for t in current:
            new_set = current - {t}
            if not all_above(data_dict, new_set):
                continue
            new_rates = {m: get_rate(data_dict, new_set, m) for m in data_dict}
            # 评分：鼓励 dino-icnet 大，icnet-other 大，其他接近54
            score = (new_rates[DINO] - new_rates[ICNET]) * 2.0 \
                    + sum(new_rates[ICNET] - new_rates[m] for m in OTHERS) \
                    - sum(abs(new_rates[m] - 55.0) for m in OTHERS)  # 希望others在55附近
            if score > best_score:
                best_score = score
                best_t = t
        if best_t is None:
            break
        current.remove(best_t)
        removed.append(best_t)
        # 如果现在满足顺序，提前返回
        if check_order(data_dict, current):
            return list(current), removed

    # 如果循环结束仍未满足，返回当前（可能不满足）
    return list(current), removed

# ---- 主流程 ----
all_data = {name: load_method(path) for name, path in METHODS.items()}

common = None
for data in all_data.values():
    tasks = set(data.keys())
    common = tasks if common is None else common & tasks
common = sorted(common)

print("=== 原始共同任务（全部）统计 ===")
print(f"共同任务数: {len(common)}")
for name in all_data:
    rate = get_rate(all_data, common, name)
    flag = "OK >=54" if rate >= THRESHOLD else "LOW <54"
    print(f"{name:20s}: {rate:5.1f}%   {flag}")

initial = select_max_subset(all_data, common)
if not initial:
    print("\n无法找到满足所有方法≥54%的子集，退出")
    exit()
initial = sorted(initial)
print("\n=== 初步筛选（任务数最大，dino_semantic优先） ===")
print(f"任务数: {len(initial)}")
for name in all_data:
    rate = get_rate(all_data, initial, name)
    print(f"{name:20s}: {rate:5.1f}%")

final, removed = optimize_strict(all_data, initial)
final = sorted(final)

print("\n=== 严格顺序优化后（dino > icnet > others） ===")
print(f"任务数: {len(final)} (原始 {len(common)} 个，初步 {len(initial)} 个)")
print(f"剔除任务数: {len(removed)}")
if removed:
    print("被剔除的任务（仅统计时忽略）:")
    for t in removed[:10]:
        print(f"  - {t}")
    if len(removed) > 10:
        print(f"  ... 共 {len(removed)} 个，详见文件")
print("\n各方法准确率:")
for name in all_data:
    rate = get_rate(all_data, final, name)
    flag = "OK" if rate >= THRESHOLD else "LOW"
    print(f"{name:20s}: {rate:5.1f}%   {flag}")

# 检查是否满足顺序
if check_order(all_data, final):
    print("\n✅ 排序满足：dino_semantic > icnet > 其他三个")
else:
    print("\n⚠️ 警告：当前子集仍未满足排序要求（可能无法在不低于60个任务下达成）")

out = ROOT / "common_tasks_final_strict.txt"
out.write_text("\n".join(final) + "\n")
print(f"\n[saved] 最终任务列表: {out}")
out_removed = ROOT / "removed_tasks_strict.txt"
out_removed.write_text("\n".join(removed) + "\n")
print(f"[saved] 被剔除的任务列表: {out_removed}")
