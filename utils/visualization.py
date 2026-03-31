import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def set_grid_ticks(ax, rows, cols, step=50):
    """统一坐标轴刻度与范围锁定"""
    ax.set_xticks(np.arange(0, cols + step, step))
    ax.set_yticks(np.arange(0, rows + step, step))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

def plot_map_3d(Z, title='', save_path=None):
    """绘制 3D 地形/建筑高程图"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.view_init(30, -37.5)
    ax.set_title(title, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_map_with_nodes(elevation, demand, supply, title='', save_path=None):
    """绘制节点分布图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    # 背景底色
    ax.imshow(elevation != 0, cmap=ListedColormap(['white', '#E6E6FA']))
    # 需求点 (蓝色)
    ax.scatter(demand[:, 2], demand[:, 1], c='blue', s=20, alpha=0.6, label='Demand Points')
    # 配送点 (红色方块)
    ax.scatter(supply[:, 2], supply[:, 1], c='red', marker='s', s=80, edgecolors='black', label='Supply Centers')
    set_grid_ticks(ax, *elevation.shape)
    ax.legend()
    ax.set_title(title, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_binary_matrix(matrix, title='', save_path=None):
    """绘制 0-1 二值矩阵图（如可行域）"""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(matrix, cmap=ListedColormap(['white', '#0072BD']))
    set_grid_ticks(ax, *matrix.shape)
    ax.set_title(title, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_score(score_matrix, title='', cmap_name='hot', save_path=None):
    """绘制评分热力图"""
    fig, ax = plt.subplots(figsize=(10, 7))
    # 屏蔽 0 分区域，使底色干净
    plot_data = np.where(score_matrix == 0, np.nan, score_matrix)
    im = ax.imshow(plot_data, cmap=cmap_name)
    set_grid_ticks(ax, *score_matrix.shape)
    plt.colorbar(im, label='Score')
    ax.set_title(title, fontweight='bold')
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_grid_selection(valid_judging, selected_points, N, title='', save_path=None):
    """绘制降采样精英点筛选结果"""
    fig, ax = plt.subplots(figsize=(10, 8))
    rows, cols = valid_judging.shape
    
    # 1. 绘制底色和所有可行格点
    ax.imshow(valid_judging == 0, cmap='Greys', alpha=0.05) 
    all_r, all_c = np.where(valid_judging == 1)
    ax.scatter(all_c, all_r, s=1, c='lightgray', label='Viable Area')
    
    # 2. 绘制筛选出的红点
    ax.scatter(selected_points[:, 1], selected_points[:, 0], s=12, c='red', label='Elite Candidates')
    
    # 3. 绘制大网格辅助线
    for r in range(0, rows, N):
        ax.axhline(r, color='blue', linestyle='--', linewidth=0.3, alpha=0.2)
    for c in range(0, cols, N):
        ax.axvline(c, color='blue', linestyle='--', linewidth=0.3, alpha=0.2)

    ax.set_title(title, fontweight='bold')
    set_grid_ticks(ax, rows, cols)
    ax.legend(loc='upper right')
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)