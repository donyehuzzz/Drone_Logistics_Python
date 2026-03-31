import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

# 全局字体设置，防止图表中文显示方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def set_grid_ticks(ax, rows, cols, step=50):
    """辅助函数：强制将坐标轴刻度锁定为指定的步长 (默认 50)"""
    ax.set_xticks(np.arange(0, cols + step, step))
    ax.set_yticks(np.arange(0, rows + step, step))
    # 限制显示范围，防止越界
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5) # imshow 默认 y 轴向下递增

def plot_map_3d(Z, title='3D Elevation Map', xlabel='Col (Grid)', ylabel='Row (Grid)', zlabel='Elevation (m)', save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    rows, cols = Z.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='m')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    
    # 设置 3D 轴的刻度
    ax.set_xticks(np.arange(0, cols + 50, 50))
    ax.set_yticks(np.arange(0, rows + 50, 50))
    
    ax.view_init(elev=30, azim=-37.5)
    ax.invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig) # 关键：关闭内存中的图形，彻底解决阻塞！

def plot_map_with_nodes(elevation, demand_points, supply_points, title='Map with Nodes', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    rows, cols = elevation.shape
    
    binary_matrix = (elevation != 0).astype(int)
    cmap = ListedColormap(['white', '#E6E6FA']) 
    ax.imshow(binary_matrix, cmap=cmap)
    
    ax.scatter(demand_points[:, 2], demand_points[:, 1], c='blue', s=30, edgecolors='k', label='需求点', zorder=2)
    ax.scatter(supply_points[:, 2], supply_points[:, 1], c='red', marker='s', s=80, edgecolors='k', label='配送中心', zorder=3)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Col (Grid)')
    ax.set_ylabel('Row (Grid)')
    
    set_grid_ticks(ax, rows, cols) # 强制 50 步长刻度
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_binary_matrix(binary_matrix, title='Candidate Sites', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    rows, cols = binary_matrix.shape
    
    cmap = ListedColormap(['white', '#0072BD']) 
    ax.imshow(binary_matrix, cmap=cmap)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Col (Grid)')
    ax.set_ylabel('Row (Grid)')
    
    set_grid_ticks(ax, rows, cols)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    import matplotlib.patches as mpatches
    patch1 = mpatches.Patch(color='#0072BD', label='Viable Area')
    patch0 = mpatches.Patch(color='white', label='Non-viable Area')
    ax.legend(handles=[patch0, patch1], loc='best')
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)

def plot_score(score_matrix, title='Score Heatmap', cmap_name='hot', save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    rows, cols = score_matrix.shape
    
    plot_data = np.where(score_matrix == 0, np.nan, score_matrix)
    ax.set_facecolor('white')
    im = ax.imshow(plot_data, cmap=cmap_name)
    fig.colorbar(im, ax=ax, label='Value', fraction=0.046, pad=0.04)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Col (Grid)')
    ax.set_ylabel('Row (Grid)')
    
    set_grid_ticks(ax, rows, cols)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)