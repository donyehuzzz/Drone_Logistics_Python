import numpy as np
from scipy.signal import convolve2d

def find_candidate_sites(type_matrix, N):
    """
    对应 MATLAB: find_candidate_sites.m
    筛选满足 N 阶摩尔邻域约束的有效备选起降点 (卷积极速版)
    """
    if not isinstance(N, int) or N < 1:
        raise ValueError(f"摩尔邻域阶数 N 必须是正整数！当前输入：{N}")

    # 1. 生成初始的空白单元格掩码 (1表示空白，0表示非空白)
    blank_mask = (type_matrix == 9).astype(int)
    
    # 获取空白单元格的线性一维索引 (对应 MATLAB 的 find)
    blank_indices_linear = np.flatnonzero(blank_mask)
    num_blank_units = len(blank_indices_linear)

    # ===================== 核心优化：二维卷积替代邻域遍历 =====================
    window_size = 2 * N + 1
    kernel = np.ones((window_size, window_size))

    # 使用 convolve2d 计算每个单元格邻域内的空白网格数量
    # mode='same' 保证输出矩阵与原矩阵大小一致，boundary='fill', fillvalue=0 等价于边缘越界补 0
    neighbourhood_sum = convolve2d(blank_mask, kernel, mode='same', boundary='fill', fillvalue=0)

    # 如果一个网格的邻域和等于窗口总面积，说明它周围全都是9
    target_sum = window_size ** 2
    valid_vertiports_judging = (neighbourhood_sum == target_sum).astype(float)

    # 2. 从判断矩阵中提取坐标
    valid_rows, valid_cols = np.where(valid_vertiports_judging == 1)
    # 将行和列拼成 [numCandidates, 2] 的坐标矩阵
    valid_vertiports_coords = np.column_stack((valid_rows, valid_cols))
    num_valid_vertiports = len(valid_rows)

    # 3. 找到有效点在“初始空白单元格列表”中的相对索引
    # flatten() 将二维矩阵拉平，方便通过一维索引查询
    is_valid_in_blank_list = valid_vertiports_judging.flatten()[blank_indices_linear]
    valid_vertiports_indices = np.where(is_valid_in_blank_list == 1)[0]

    print(f"   - 地图尺寸: {type_matrix.shape[0]} x {type_matrix.shape[1]}")
    print(f"   - 满足 {N} 阶摩尔邻域约束的备选起降点：{num_valid_vertiports}个 (由 {num_blank_units} 个初始空白区中筛选)")

    return valid_vertiports_coords, valid_vertiports_indices, valid_vertiports_judging