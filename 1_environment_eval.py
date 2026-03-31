import os
import time
import numpy as np

# 导入自定义模块
from utils.map_tools import read_map_grids, process_logistics_nodes
from utils.vertiport_tools import find_candidate_sites
from utils.evaluation_models import calc_noise_score, calc_convenience_score
from utils.visualization import plot_map_3d, plot_map_with_nodes, plot_binary_matrix, plot_score

def main():
    print("="*50)
    print("🚀 启动城市低空物流起降点网络 - 环境评价与筛选引擎")
    print("="*50)
    start_time = time.time()

    # ---------------------------------------------------------
    # 0. 全局参数配置区
    # ---------------------------------------------------------
    DATA_DIR = "data"
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True) # 自动创建输出文件夹

    # 对应 MATLAB 里的文件路径
    MAP_PATHS = [os.path.join(DATA_DIR, '区域高程.xlsx'), os.path.join(DATA_DIR, '区域类型.xlsx')]
    NODE_PATHS = [os.path.join(DATA_DIR, '需求点.xlsx'), os.path.join(DATA_DIR, '配送点.xlsx')] 

    # 物理参数
    RADIUS_SP = [200, 200]
    N_MOORE = 2
    NOISE_PARAMS = [120, 0.1, 11, 30, 200] # s_uav, alpha, s_reflect, h_u, r_noi
    CONV_PARAMS = [1, 180, 5]             # v_p, T_p, grid_size
    WEIGHTS = [0.5, 0.5]                  # 噪声权重, 便利性权重

    # ---------------------------------------------------------
    # 1. 数据预处理
    # ---------------------------------------------------------
    print("\n>>> 正在读取地图与节点数据...")
    elevation, type_matrix = read_map_grids(MAP_PATHS)
    demand_points_info, supply_points_info, OD_ds = process_logistics_nodes(NODE_PATHS, RADIUS_SP)

    print("\n>>> 正在后台静默渲染高清原图 (存放至 output/ 文件夹)...")
    plot_map_3d(elevation, title="3D Terrain & Building Elevation", 
                save_path=os.path.join(OUTPUT_DIR, "Fig1_3D_Elevation.png"))
    plot_map_with_nodes(elevation, demand_points_info, supply_points_info, title="Nodes Distribution", 
                        save_path=os.path.join(OUTPUT_DIR, "Fig2_Nodes_Distribution.png"))

    # ---------------------------------------------------------
    # 2. 起降点评估
    # ---------------------------------------------------------
    print("\n>>> 开始进行选址环境评价...")
    
    # 2.1 筛选满足约束的合格起降点
    valid_coords, _, valid_judging = find_candidate_sites(type_matrix, N_MOORE)

    # 2.2 评价得分计算
    print("    -> 计算噪声与便利性得分...")
    noise_score = calc_noise_score(valid_coords, type_matrix, *NOISE_PARAMS)
    conv_score = calc_convenience_score(valid_judging, demand_points_info, *CONV_PARAMS)

    # 2.3 综合评价
    print("    -> 计算综合评价得分...")
    comprehensive_score = (noise_score * WEIGHTS[0] + conv_score * WEIGHTS[1]) * valid_judging

    # ---------------------------------------------------------
    # 3. 评价结果可视化
    # ---------------------------------------------------------
    print("\n>>> 正在后台静默渲染评价热力图...")
    plot_binary_matrix(valid_judging, title="Viable Candidate Vertiports", 
                       save_path=os.path.join(OUTPUT_DIR, "Fig3_Valid_Candidates.png"))
    plot_score(noise_score, title="Noise Evaluation Score", 
               save_path=os.path.join(OUTPUT_DIR, "Fig4_Noise_Score.png"))
    plot_score(conv_score, title="Convenience Evaluation Score", 
               save_path=os.path.join(OUTPUT_DIR, "Fig5_Conv_Score.png"))
    plot_score(comprehensive_score, title="Comprehensive Suitability Score", 
               save_path=os.path.join(OUTPUT_DIR, "Fig6_Comp_Score.png"))

    # ---------------------------------------------------------
    # 4. 结果导出 (放入 output 文件夹)
    # ---------------------------------------------------------
    save_path = os.path.join(OUTPUT_DIR, "01_Results_of_vtpsEval.npz")
    np.savez(save_path, elevation=elevation, type_matrix=type_matrix, 
             demand_points_info=demand_points_info, supply_points_info=supply_points_info, 
             OD_ds=OD_ds, valid_judging=valid_judging, noise_score=noise_score, 
             conv_score=conv_score, comprehensive_score=comprehensive_score, valid_coords=valid_coords)

    print(f"\n🎉 运行完成！总耗时: {time.time() - start_time:.2f} 秒")
    print(f"✅ 核心资产已保存至: {save_path}")

if __name__ == "__main__":
    main()