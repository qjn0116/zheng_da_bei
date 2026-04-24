# 🧸 Z世代娃衣消费行为分析

> 正大杯参赛项目 · 数据分析师求职作品集

## 📌 项目简介

本项目聚焦 Z 世代棉花娃娃爱好者，围绕「身份认同 — 情感补偿 — 购买行为」三维框架，基于 **400 份问卷 + 15 场深度访谈 + 200+ 社交媒体评论**，系统分析了娃衣消费的心理动因、行为特征与群体分化规律。

**核心结论：**
- 娃衣消费呈「高频次、低金额、强冲动」特征，月消费 101-500 元占比 **75.5%**
- 审美是第一驱动力，「独特设计」得分远超「价格合理性」
- 消费者可划分为 4 类：情感驱动型（34.2%）、深度社群型（22.8%）、高消费中情感型（19.2%）、浅尝辄止型（23.8%）

---

## 🛠️ 技术架构

| 层级 | 技术 | 说明 |
| :--- | :--- | :--- |
| **云基础设施** | 阿里云 RDS MySQL 8.0 | 云端数据存储，模拟真实生产环境 |
| **数据探查** | SQL (DataGrip) | 数据清洗、分组统计、窗口函数、特征工程 |
| **数据分析** | Python (Pandas, SciPy, Scikit-learn) | 描述统计、差异检验、聚类、决策树 |
| **大模型** | DeepSeek API | 用户评论情感分类、主题提取、交叉验证 |
| **可视化** | Tableau / Matplotlib / Seaborn | 交互看板、雷达图、热力图 |
| **项目管理** | Git + GitHub | 版本控制、代码托管、项目展示 |

---

## 📁 文件结构
zheng_da_bei/
├── README.md # 项目说明（本文件）
├── data/
│ ├── raw_data.csv # 脱敏后的问卷数据（100条样例）
│ └── data_dictionary.md # 数据字典
├── sql/
│ └── analysis.sql # SQL 探查脚本（含结果注释）
├── notebooks/
│ ├── 01_eda_analysis.ipynb # 描述统计 + 可视化
│ ├── 02_clustering.ipynb # K-Means 聚类分析
│ └── 03_decision_tree.ipynb # 决策树（购买意愿预测）
├── llm/
│ └── sentiment_analysis.ipynb # 大模型情感分析
├── dashboard/
│ └── dashboard_link.txt # Tableau 看板链接
└── report/
└── 正大杯项目报告.pdf # 原始论文


---

## 💾 数据说明

- **数据来源**：问卷星（400 份有效问卷）+ 深度访谈（15 人）+ 社交媒体评论（210 条）
- **脱敏处理**：已删除 IP 地址、用户标识等个人信息
- **存储方式**：阿里云 RDS MySQL，支持远程访问和 SQL 探查

> 详细字段说明见 `data/data_dictionary.md`

---

## 🗄️ SQL 分析（阿里云 RDS + DataGrip）

### 数据探查
```sql
-- 总样本量：400 份
SELECT COUNT(*) FROM survey_data;

-- 年龄分布：19-26岁占 75%
SELECT age_group, COUNT(*) FROM survey_data GROUP BY age_group;
消费行为分析
sql
-- 月消费分布：101-500元占 75.5%
SELECT monthly_expense, COUNT(*) FROM survey_data GROUP BY monthly_expense;

-- 冲动购买倾向：70% 用户同意/非常同意
SELECT impulse_buy, COUNT(*) FROM survey_data GROUP BY impulse_buy;
分组统计与窗口函数
sql
-- 不同收入组的冲动购买倾向
SELECT monthly_income, AVG(impulse_buy_score) FROM survey_data GROUP BY monthly_income;

-- 按年龄组对消费金额排序
SELECT age_group, monthly_expense, ROW_NUMBER() OVER (PARTITION BY age_group ORDER BY expense_level DESC) FROM survey_data;
> 完整 SQL 脚本见 sql/analysis.sql

---
##📈 Python 分析
待补充（Notebook 完成后更新）

计划内容
描述性统计 + 可视化

差异检验（年龄/收入对消费的影响）

相关性分析 + 热力图

K-Means 聚类（4 类用户画像）

决策树（购买意愿预测 + 特征重要性）

---
##🤖 AI 大模型验证
待补充（调用 DeepSeek API 分析用户评论）

计划内容
对 200+ 社交媒体评论进行情感分类

提取主题：审美 / 情感补偿 / 社交 / 收藏

与问卷分析结论交叉验证

---
##📊 可视化看板（Tableau）
待补充

[点击查看交互看板]（链接待补充）

---
##📝 项目亮点
亮点	说明
真实数据 + 云端存储	400 份真实问卷，数据存储在阿里云 RDS，模拟企业级数据流
全链路技术覆盖	SQL → Python → AI → 可视化，完整闭环
多元化分析方法	描述统计、聚类、决策树、大模型情感分析
业务导向结论	用户分群、转化路径、精准营销策略，可直接落地
版本控制规范	GitHub 托管，SQL 脚本、Notebook、文档结构清晰
🔗 相关链接
[阿里云 RDS 控制台]（内部）

[Tableau 看板]（待补充）

[正大杯原报告]（report/）


###👤 作者
GitHub：qjn0116

项目时间：2025-2026

联系方式：待补充

📄 License
仅供学习交流使用。数据已脱敏，不涉及个人隐私。
