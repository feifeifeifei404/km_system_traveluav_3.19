# src/vlnce_src/scoring_util.py

import numpy as np

# --- 权重超参数 ---
W_OBSTACLE = 0.5
W_TARGET = 0.3
W_PROGRESS = 0.2

def _calculate_obstacle_score(candidate_endpoint, current_episode):
    """
    计算障碍物分数。分数越高越安全。
    """
    depth_maps = current_episode[-1]['depth']
    relevant_depths = [depth_maps[0], depth_maps[1], depth_maps[2], depth_maps[4]]
    min_depths = []
    for depth_map in relevant_depths:
        h, w = depth_map.shape
        center_h, center_w = h // 2, w // 2
        crop_size = min(h, w) // 4
        center_crop = depth_map[center_h - crop_size : center_h + crop_size, 
                                center_w - crop_size : center_w + crop_size]
        min_depths.append(np.min(center_crop) if center_crop.size > 0 else 0)
    safety_distance = min(min_depths)
    score = np.log(safety_distance + 1)
    return score

def _calculate_target_score(candidate_endpoint, current_pos, target_pos):
    """
    计算目标朝向分数。
    """
    vec_to_candidate = np.array(candidate_endpoint) - np.array(current_pos)
    vec_to_target = np.array(target_pos) - np.array(current_pos)
    norm_candidate = np.linalg.norm(vec_to_candidate)
    norm_target = np.linalg.norm(vec_to_target)
    if norm_candidate < 1e-6 or norm_target < 1e-6:
        return 0.0
    cosine_similarity = np.dot(vec_to_candidate, vec_to_target) / (norm_candidate * norm_target)
    return (cosine_similarity + 1) / 2

def _calculate_progress_score(candidate_endpoint, current_pos):
    """
    计算进展分数。
    """
    distance = np.linalg.norm(np.array(candidate_endpoint) - np.array(current_pos))
    return np.tanh(distance / 10.0)

def score_and_select_best_waypoint(candidates: list, current_episode: list, target_position: list):
    """
    对所有候选路径进行评分，并选出最优的一个。
    """
    if not candidates:
        return None

    scores = []
    current_pos = current_episode[-1]['sensors']['state']['position']

    for candidate_path in candidates:
        # --- 核心改动：只取路径的最后一个点作为评估对象 ---
        endpoint = candidate_path[-1]

        # 用这个终点来计算所有分数
        s_obstacle = _calculate_obstacle_score(endpoint, current_episode)
        s_target = _calculate_target_score(endpoint, current_pos, target_position)
        s_progress = _calculate_progress_score(endpoint, current_pos)
        
        total_score = (W_OBSTACLE * s_obstacle +
                       W_TARGET * s_target +
                       W_PROGRESS * s_progress)
        scores.append(total_score)

    if not scores:
        return candidates[0] # 终极备用

    best_index = np.argmax(scores)
    # 返回的是完整的最佳路径，而不仅仅是终点
    best_path = candidates[best_index]
    
    return best_path