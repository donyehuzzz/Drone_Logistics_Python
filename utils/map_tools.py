import pandas as pd
import numpy as np
import warnings

def read_map_grids(map_info_paths):
    """
    对应 MATLAB: read_map_grids.m
    读取高程和土地类型矩阵，严格限制范围防止 Excel "幽灵列"
    """
    path1, path2 = map_info_paths[0], map_info_paths[1]
    
    # 完美对标 MATLAB 的 'Range', 'A2:NP325'
    # usecols="A:NP" 保证只读到第 380 列
    # nrows=324 保证只读 324 行
    # skiprows=1 相当于跳过了第一行表头，从第 2 行开始读
    
    try:
        elevation = pd.read_excel(path1, skiprows=1, header=None, usecols="A:NP", nrows=324).values
        type_matrix = pd.read_excel(path2, skiprows=1, header=None, usecols="A:NP", nrows=324).values
    except Exception as e:
        print(f"❌ 读取地图矩阵失败... 错误: {e}")
        # 如果报错，说明可能没有表头，我们去掉 skiprows 重试，但依然死死限制范围！
        elevation = pd.read_excel(path1, header=None, usecols="A:NP", nrows=324).values
        type_matrix = pd.read_excel(path2, header=None, usecols="A:NP", nrows=324).values

    # 暴力清洗：把读取过程中可能产生的任何 NaN (空白单元格) 强制转换为 0
    elevation = np.nan_to_num(elevation, nan=0.0)
    type_matrix = np.nan_to_num(type_matrix, nan=0.0)

    print(f"   - ✅ 地图基础数据读取完成，严格截断后维度: {elevation.shape}")
    return elevation, type_matrix

def process_logistics_nodes(node_coords_paths, radius_sp):
    """
    对应 MATLAB: process_logistics_nodes.m
    处理需求点和配送中心，并计算 OD 矩阵 (向量化极速版)
    """
    path1, path2 = node_coords_paths[0], node_coords_paths[1]

    # ===================== 第一部分：需求点处理 =====================
    # MATLAB 中读取了 A2:D122。Python 中我们直接读取前 4 列，并跳过表头
    df_demand = pd.read_excel(path1, usecols="A:D", skiprows=1, header=None)
    demand_points_info = df_demand.values
    num_demand_point = demand_points_info.shape[0]
    print('   - 需求点数据已直接读取完成')

    # ===================== 第二部分：配送中心+OD矩阵处理 =====================
    df_supply = pd.read_excel(path2, usecols="A:C", skiprows=1, header=None)
    
    # Python 数组拼接：为了和 MATLAB 保持一致，初始化 4 列，第 4 列用于存总货量
    supply_points_info = np.zeros((df_supply.shape[0], 4))
    supply_points_info[:, 0:3] = df_supply.values
    
    # 初始化 OD 矩阵 [k x 2]
    k = num_demand_point
    OD_ds = np.zeros((k, 2))

    # 批量提取坐标和需求量 (注意：Python 索引从 0 开始，所以 MATLAB 的 2,3,4 对应 Python 的 1,2,3)
    d_rows = demand_points_info[:, 1]
    d_cols = demand_points_info[:, 2]
    total_demands = demand_points_info[:, 3]

    s1_row, s1_col = supply_points_info[0, 1], supply_points_info[0, 2]
    s2_row, s2_col = supply_points_info[1, 1], supply_points_info[1, 2]

    # 一次性计算所有需求点到两个配送中心的直线距离向量
    s1_dist = np.sqrt((s1_row - d_rows)**2 + (s1_col - d_cols)**2)
    s2_dist = np.sqrt((s2_row - d_rows)**2 + (s2_col - d_cols)**2)

    # 生成布尔逻辑索引
    in_s1 = s1_dist <= radius_sp[0]
    in_s2 = s2_dist <= radius_sp[1]

    # 场景A: 同时在两个配送中心范围内
    idx_both = in_s1 & in_s2
    if np.any(idx_both):
        sum_dist = s1_dist[idx_both] + s2_dist[idx_both]
        
        # 计算比例 (防除0保护)
        ratio1 = s2_dist[idx_both] / sum_dist
        ratio2 = s1_dist[idx_both] / sum_dist
        
        # 如果恰好距离之和为0（两个中心重合），平分
        ratio1[sum_dist == 0] = 0.5
        ratio2[sum_dist == 0] = 0.5
        
        OD_ds[idx_both, 0] = total_demands[idx_both] * ratio1
        OD_ds[idx_both, 1] = total_demands[idx_both] * ratio2

    # 场景B: 仅在配送中心1范围内
    idx_s1_only = in_s1 & ~in_s2
    OD_ds[idx_s1_only, 0] = total_demands[idx_s1_only]

    # 场景C: 仅在配送中心2范围内
    idx_s2_only = ~in_s1 & in_s2
    OD_ds[idx_s2_only, 1] = total_demands[idx_s2_only]

    # 场景D: 均不在范围内
    idx_none = ~in_s1 & ~in_s2
    if np.any(idx_none):
        # 提取未覆盖的需求点 ID (第 0 列)
        uncovered_ids = demand_points_info[idx_none, 0]
        warnings.warn(f"共有 {np.sum(idx_none)} 个需求点不在任何配送中心的服务范围内，OD值设为0！\n(ID列表: {uncovered_ids.tolist()})")

    # 4、计算每个配送中心的总运输量并存入矩阵 (第 3 列)
    supply_points_info[0, 3] = np.sum(OD_ds[:, 0])
    supply_points_info[1, 3] = np.sum(OD_ds[:, 1])

    print('   - 配送中心--需求点OD矩阵 (极速版) 已计算完成')

    return demand_points_info, supply_points_info, OD_ds