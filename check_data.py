import numpy as np

# 1. 加载 npz 压缩包
data_path = "data/01_Results_of_vtpsEval.npz"
data = np.load(data_path)

# 2. 查看里面装了哪些“盲盒” (也就是变量名)
print("📦 文件中包含的变量列表:")
print(data.files)

# 3. 拿几个核心数据出来看看“成色”
comp_score = data['comprehensive_score']
valid_coords = data['valid_coords']

print("\n📊 综合评价得分矩阵 (comprehensive_score):")
print(f"   -> 矩阵维度: {comp_score.shape}")
print(f"   -> 最高得分: {np.max(comp_score):.4f}")
# 挑出所有大于0的有效得分，看看最低分是多少
valid_scores = comp_score[comp_score > 0]
if len(valid_scores) > 0:
    print(f"   -> 最低有效得分: {np.min(valid_scores):.4f}")

print("\n📍 合格起降点坐标 (valid_coords):")
print(f"   -> 备选点总数: {valid_coords.shape[0]} 个")
print(f"   -> 前 5 个坐标 [行, 列]:\n{valid_coords[:5]}")

# 4. 养成好习惯，看完关上箱子
data.close()

# %%
import numpy as np
data = np.load("data/01_Results_of_vtpsEval.npz")
comp_score = data['comprehensive_score']