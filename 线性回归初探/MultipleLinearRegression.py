import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  sklearn.linear_model import LinearRegression
# 读取生活满意度csv

bli = pd.read_csv("BLI_14012020152147151.csv")
# 过滤出INEQUALITY为TOT的数据
bli = bli[bli["INEQUALITY"] == "TOT"]
# 设置行列
bli = bli.pivot(index="Country", columns="Indicator", values="Value")
# 读取GDP数据
weo = pd.read_csv("WEO_Data.csv", encoding="unicode_escape")
weo.rename(columns={"2015": "GDP_PRE_CAPITA"}, inplace=True)
weo = weo.set_index("Country")

# 合并两个数据集
country_stats = pd.merge(left=bli, right=weo, left_index=True, right_index=True)
country_stats = country_stats[["GDP_PRE_CAPITA", "Life satisfaction"]]
country_stats.sort_values(by="GDP_PRE_CAPITA", inplace=True)

# 可视化
country_stats.plot(kind='scatter', x="GDP_PRE_CAPITA", y='Life satisfaction')
plt.show()

# 保存到csv
country_stats.to_csv("country_stats.csv")

# 清理异常值
remove_indices = [0, 39]
keep_indices = list(set(range(40)) - set(remove_indices))

sample_data = country_stats[["GDP_PRE_CAPITA", "Life satisfaction"]].iloc[keep_indices]
sample_data.plot(kind='scatter', x="GDP_PRE_CAPITA", y='Life satisfaction')
plt.show()

# 选择模型
model = LinearRegression()

# 训练模型
X = np.c_[country_stats["GDP_PRE_CAPITA"]]
y = np.c_[country_stats["Life satisfaction"]]
model.fit(X, y)

# 模型预测
X_new = [[22587]]
print(model.predict(X_new))
