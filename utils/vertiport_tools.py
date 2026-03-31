import numpy as np
import pandas as pd
from scipy.signal import convolve2d

def find_candidate_sites(type_matrix, N):
    """【Step 1】N 阶摩尔邻域卷积筛选：找出所有物理上可行（周围全为空白区）的网格"""
    blank_mask = (type_matrix == 9).astype(int)
    kernel = np.ones((2 * N + 1, 2 * N + 1))
    # 只有当中心点周围 (2N+1)x(2N+1) 全为 9 时，该点才可行
    neighbourhood_sum = convolve2d(blank_mask, kernel, mode='same')
    valid_judging = (neighbourhood_sum == kernel.size).astype(float)
    
    valid_rows, valid_cols = np.where(valid_judging == 1)
    valid_coords = np.column_stack((valid_rows, valid_cols))
    return valid_coords, valid_judging

def select_grid_top_k(valid_judging, comprehensive_score, k, N_grid):
    """【Step 2】空间降采样：在 N_grid x N_grid 的大网格中挑选得分最高的 K 个点"""
    rows, cols = np.where(valid_judging == 1)
    scores = comprehensive_score[rows, cols]
    
    if len(rows) == 0:
        return None

    # 计算该点属于哪个大网格 (Python 0-indexed)
    grid_r = rows // N_grid
    grid_c = cols // N_grid

    # 利用 Pandas 进行快速分组排序
    df = pd.DataFrame({
        'row': rows, 'col': cols, 'score': scores,
        'grid_r': grid_r, 'grid_c': grid_c
    })

    # 先按网格 ID 排序，网格内按得分降序，取前 K 个
    selected_df = df.sort_values(by=['grid_r', 'grid_c', 'score'], ascending=[True, True, False])
    selected_df = selected_df.groupby(['grid_r', 'grid_c']).head(k)

    # 返回结果包含：[行, 列, 得分, 网格行ID, 网格列ID]
    return selected_df.values