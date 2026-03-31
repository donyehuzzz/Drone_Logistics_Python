import os
import time
import numpy as np
from scipy.spatial.distance import cdist

# 导入自定义模块
from utils.map_tools import read_map_grids, process_logistics_nodes
from utils.vertiport_tools import find_candidate_sites, select_grid_top_k
from utils.evaluation_models import calc_noise_score, calc_convenience_score
from utils.visualization import (plot_map_3d, plot_map_with_nodes, 
                                plot_binary_matrix, plot_score, plot_grid_selection)

def main():
    print("="*60)
    print("🚀 启动低空物流起降点预处理引擎 (环境评价 + 精英筛选)")
    print("="*60)
    start_time = time.time()

    # ---------------------------------------------------------
    # 0. 参数配置
    # ---------------------------------------------------------
    DATA_DIR, OUTPUT_DIR = "data", "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 对应 Excel 路径
    MAP_PATHS = [os.path.join(DATA_DIR, '区域高程.xlsx'), os.path.join(DATA_DIR, '区域类型.xlsx')]
    NODE_PATHS = [os.path.join(DATA_DIR, '需求点.xlsx'), os.path.join(DATA_DIR, '配送点.xlsx')] 

    # 算法参数
    N_MOORE = 2           # 摩尔邻域阶数
    GRID_N, GRID_K = 10, 1 # 空间降采样：10x10网格内选Top 1
    WEIGHTS = [0.5, 0.5]  # [噪声权重, 便利性权重]

    # ---------------------------------------------------------
    # 1. 数据读取与基础图件
    # ---------------------------------------------------------
    print("\n>>> 正在加载基础地理与物流数据...")
    elevation, type_matrix = read_map_grids(MAP_PATHS)
    demand_info, supply_info, OD_ds = process_logistics_nodes(NODE_PATHS, [200, 200])

    # ---------------------------------------------------------
    # 2. 环境评估 (Step 01 逻辑)
    # ---------------------------------------------------------
    print("\n>>> 正在计算环境评价模型...")
    # 筛选物理可行点
    _, valid_judging = find_candidate_sites(type_matrix, N_MOORE)
    
    # 计算各项得分
    noise_score = calc_noise_score(np.column_stack(np.where(valid_judging==1)), type_matrix, 120, 0.1, 11, 30, 200)
    conv_score = calc_convenience_score(valid_judging, demand_info, 1, 180, 5)
    
    # 综合评分 (强制锁定在可行点范围内)
    comp_score = (noise_score * WEIGHTS[0] + conv_score * WEIGHTS[1]) * valid_judging

    # ---------------------------------------------------------
    # 3. 精英点筛选与距离预计算 (Step 02 逻辑)
    # ---------------------------------------------------------
    print("\n>>> 正在执行空间降采样 (Grid Top-K)...")
    selected_points = select_grid_top_k(valid_judging, comp_score, GRID_K, GRID_N)
    
    print(">>> 正在预计算空间距离矩阵 (欧式/曼哈顿)...")
    demand_coords = demand_info[:, 1:3] * 5      # 转为米
    candidate_coords = selected_points[:, 0:2] * 5
    
    dist_dc_euc = cdist(demand_coords, candidate_coords, 'euclidean')
    dist_dc_man = cdist(demand_coords, candidate_coords, 'cityblock')
    dist_cc_euc = cdist(candidate_coords, candidate_coords, 'euclidean')

    # ---------------------------------------------------------
    # 4. 可视化与资产导出
    # ---------------------------------------------------------
    print("\n>>> 正在生成高清成果图件...")
    plot_score(comp_score, title="Comprehensive Suitability Score", 
               save_path=os.path.join(OUTPUT_DIR, "Fig1_Comp_Score.png"))
    plot_grid_selection(valid_judging, selected_points, GRID_N, title="Elite Candidate Sites", 
                        save_path=os.path.join(OUTPUT_DIR, "Fig2_Grid_Selection.png"))

    # 最终打包 (直接对接第二阶段 NSGA-II)
    save_path = os.path.join(OUTPUT_DIR, "Final_Prep_Data.npz")
    np.savez(save_path, 
             demand_points_info=demand_info,
             selected_points=selected_points[:, 0:3], # [Row, Col, Score]
             dist_dc_euc=dist_dc_euc,
             dist_dc_man=dist_dc_man,
             dist_cc_euc=dist_cc_euc,
             elevation=elevation)

    print(f"\n🎉 全流程处理完毕！总耗时: {time.time() - start_time:.2f}s")
    print(f"📍 最终精英候选点数: {len(selected_points)}")
    print(f"📂 资产文件已就绪: {save_path}")

if __name__ == "__main__":
    main()