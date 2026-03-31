import numpy as np
from scipy.spatial.distance import cdist

def calc_noise_score(valid_vertiports_coords, type_matrix, s_uav, alpha_air, s_reflect, h_u, r_noi):
    """
    对应 MATLAB: calc_noise_score.m
    基于距离衰减的噪声评估逻辑
    """
    M, N_cols = type_matrix.shape 
    noiseScore = np.zeros((M, N_cols))  
    grid_size = 5                   
    
    # 1. 预计算悬停对地噪声 S_g (np.finfo(float).eps 等价于 MATLAB 的 eps 防止 log(0))
    h_u = max(h_u, np.finfo(float).eps)  
    S_g = s_uav - 20 * np.log10(h_u) - alpha_air * h_u - s_reflect
    
    numCandidates = valid_vertiports_coords.shape[0]
    if numCandidates == 0:
        return noiseScore
        
    candidate_Sij = np.full(numCandidates, S_g)

    # 提取所有建筑物的坐标集合 (type 值为 1到6)
    b_rows, b_cols = np.where(np.isin(type_matrix, [1, 2, 3, 4, 5, 6]))
    
    if len(b_rows) > 0:
        V_coords = valid_vertiports_coords * grid_size
        B_coords = np.column_stack((b_rows, b_cols)) * grid_size
        
        # 3. 核心：使用 cdist 一次性计算所有起降点到所有建筑物的距离矩阵
        # 结果 D_matrix 尺寸为 [numCandidates x numBuildings]
        D_matrix = cdist(V_coords, B_coords, metric='euclidean')
        
        D_matrix[D_matrix > r_noi] = np.inf
        
        # np.min(..., axis=1) 等价于 MATLAB 的 min(..., [], 2)
        min_D = np.min(D_matrix, axis=1)
        valid_idx = min_D != np.inf
        
        if np.any(valid_idx):
            d_eff = np.maximum(min_D[valid_idx], np.finfo(float).eps) 
            S_max_horizontal = s_uav - 20 * np.log10(d_eff) - alpha_air * d_eff - s_reflect
            candidate_Sij[valid_idx] = np.maximum(S_max_horizontal, S_g)

    # 5. 归一化得分
    max_Sij = np.max(candidate_Sij)  
    min_Sij = S_g                 
    
    if max_Sij == min_Sij:
        scores = np.ones(numCandidates)
    else:
        scores = (max_Sij - candidate_Sij) / (max_Sij - min_Sij)
        
    scores = np.clip(scores, 0, 1) # 限制在 [0,1] 之间

    # 赋值回二维矩阵 (Python 支持直接用坐标数组进行花式索引)
    noiseScore[valid_vertiports_coords[:, 0], valid_vertiports_coords[:, 1]] = scores
    
    return noiseScore

def calc_convenience_score(valid_vertiports_judging, demand_points_info, v_p, T_p, grid_size):
    """
    对应 MATLAB: calc_convenience_score.m
    评估候选起降点的步行便利性
    """
    M, N_cols = valid_vertiports_judging.shape
    convenienceScore = np.zeros((M, N_cols))

    c_rows, c_cols = np.where(valid_vertiports_judging == 1)
    numCandidates = len(c_rows)

    if numCandidates == 0:
        return convenienceScore 

    # Python 切片，索引从 0 开始
    dp_rows = demand_points_info[:, 1]
    dp_cols = demand_points_info[:, 2]
    dp_vols = demand_points_info[:, 3] 
    
    Candidate_Coords = np.column_stack((c_rows, c_cols)) * grid_size
    Demand_Coords = np.column_stack((dp_rows, dp_cols)) * grid_size

    # 计算曼哈顿距离矩阵 (metric='cityblock')
    D_matrix = cdist(Candidate_Coords, Demand_Coords, metric='cityblock')

    Time_matrix = D_matrix / v_p
    Satisfied_matrix = (Time_matrix <= T_p).astype(float) 

    # 【核心加权逻辑】：np.dot() 矩阵乘法
    candidate_a_ij = np.dot(Satisfied_matrix, dp_vols)

    min_a = np.min(candidate_a_ij)
    max_a = np.max(candidate_a_ij)

    if max_a == min_a:
        scores = np.ones(numCandidates)
    else:
        scores = (candidate_a_ij - min_a) / (max_a - min_a)

    scores = np.clip(scores, 0, 1)

    # 向量化赋值回全局得分矩阵
    convenienceScore[c_rows, c_cols] = scores

    return convenienceScore