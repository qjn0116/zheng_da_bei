#-------------------------------------- 第一步：验证数据导入是否正确 -------------------------------------------
-- 1. 查看总行数
SELECT COUNT(*) FROM data_raw;

-- 2. 查看前10行
SELECT * FROM data_raw LIMIT 10;

-- 3. 检查各年龄组分布
SELECT age_group, COUNT(*) AS cnt
FROM data_raw
GROUP BY age_group;

-- 4. 检查性别分布
SELECT gender, COUNT(*) AS cnt
FROM data_raw
GROUP BY gender;

-- 5. 检查城市分布
SELECT city_level, COUNT(*) AS cnt
FROM data_raw
GROUP BY city_level;

# -----------------------------------------第二步：消费行为分析------------------------------------------------
-- 1. 月消费金额分布（验证"低金额、高频次"）
SELECT monthly_expense, COUNT(*) AS cnt
FROM data_raw
GROUP BY monthly_expense
ORDER BY cnt DESC;

-- 2. 冲动购买倾向分布
SELECT impulse_buy, COUNT(*) AS cnt
FROM data_raw
GROUP BY impulse_buy
ORDER BY cnt DESC;

-- 3. 社群参与频率分布
SELECT community_participation, COUNT(*) AS cnt
FROM data_raw
GROUP BY community_participation;

-- 4. 购买决策因素排名（看哪个因素得分最高）
SELECT
    AVG(CASE design_importance
        WHEN '非常不同意' THEN 1 WHEN '不同意' THEN 2
        WHEN '一般' THEN 3 WHEN '同意' THEN 4 WHEN '非常同意' THEN 5 END) AS avg_design,
    AVG(CASE texture_importance
        WHEN '非常不同意' THEN 1 WHEN '不同意' THEN 2
        WHEN '一般' THEN 3 WHEN '同意' THEN 4 WHEN '非常同意' THEN 5 END) AS avg_texture,
    AVG(CASE price_importance
        WHEN '非常不同意' THEN 1 WHEN '不同意' THEN 2
        WHEN '一般' THEN 3 WHEN '同意' THEN 4 WHEN '非常同意' THEN 5 END) AS avg_price,
    AVG(CASE limited_importance
        WHEN '非常不同意' THEN 1 WHEN '不同意' THEN 2
        WHEN '一般' THEN 3 WHEN '同意' THEN 4 WHEN '非常同意' THEN 5 END) AS avg_limited,
    AVG(CASE designer_importance
        WHEN '非常不同意' THEN 1 WHEN '不同意' THEN 2
        WHEN '一般' THEN 3 WHEN '同意' THEN 4 WHEN '非常同意' THEN 5 END) AS avg_designer
FROM data_raw;
# -------------------------------------------------第三步：分组统计（不同人群差异）-------------------------------------
-- 1. 不同年龄组的平均消费
SELECT
    age_group,
    AVG(CASE monthly_expense
        WHEN 'A. 100元以下' THEN 1
        WHEN 'B. 101-300元' THEN 2
        WHEN 'C. 301-500元' THEN 3
        WHEN 'D. 501-1000元' THEN 4
        WHEN 'E. 1000元以上' THEN 5
    END) AS avg_expense_level
FROM data_raw
GROUP BY age_group
ORDER BY avg_expense_level DESC;

-- 2. 不同收入组的冲动购买倾向
SELECT
    monthly_income,
    AVG(CASE impulse_buy
        WHEN '非常不同意' THEN 1 WHEN '不同意' THEN 2
        WHEN '一般' THEN 3 WHEN '同意' THEN 4 WHEN '非常同意' THEN 5 END) AS avg_impulse
FROM data_raw
GROUP BY monthly_income
ORDER BY avg_impulse DESC;

-- 3. 不同城市级别的消费金额
SELECT
    city_level,
    AVG(CASE monthly_expense
        WHEN 'A. 100元以下' THEN 1
        WHEN 'B. 101-300元' THEN 2
        WHEN 'C. 301-500元' THEN 3
        WHEN 'D. 501-1000元' THEN 4
        WHEN 'E. 1000元以上' THEN 5
    END) AS avg_expense
FROM data_raw
GROUP BY city_level;
# ---------------------------------------------第四步：分层变量构造---------------------------------------------
-- 创建消费层级字段（低/中/高消费）
SELECT
    *,
    CASE
        WHEN monthly_expense IN ('A. 100元以下', 'B. 101-300元') THEN '低消费'
        WHEN monthly_expense = 'C. 301-500元' THEN '中消费'
        WHEN monthly_expense IN ('D. 501-1000元', 'E. 1000元以上') THEN '高消费'
    END AS expense_segment
FROM data_raw
LIMIT 20;

-- 创建社群参与层级
SELECT
    *,
    CASE
        WHEN community_participation IN ('A. 几乎每天浏览和互动', 'B. 每周数次') THEN '高活跃'
        WHEN community_participation = 'C. 每月数次' THEN '中活跃'
        ELSE '低活跃'
    END AS activity_level
FROM data_raw
LIMIT 20;
# ---------------------------------------------------第五步：窗口函数----------------------------------------------------
-- 按年龄组对消费金额排序
SELECT
    age_group,
    monthly_expense,
    ROW_NUMBER() OVER (
        PARTITION BY age_group
        ORDER BY CASE monthly_expense
            WHEN 'A. 100元以下' THEN 1
            WHEN 'B. 101-300元' THEN 2
            WHEN 'C. 301-500元' THEN 3
            WHEN 'D. 501-1000元' THEN 4
            WHEN 'E. 1000元以上' THEN 5
        END DESC
    ) AS rank_in_age
FROM data_raw;

-- 按城市分组，查看消费金额排名
SELECT
    city_level,
    monthly_expense,
    RANK() OVER (
        PARTITION BY city_level
        ORDER BY CASE monthly_expense
            WHEN 'A. 100元以下' THEN 1
            WHEN 'B. 101-300元' THEN 2
            WHEN 'C. 301-500元' THEN 3
            WHEN 'D. 501-1000元' THEN 4
            WHEN 'E. 1000元以上' THEN 5
        END DESC
    ) AS rank_in_city
FROM data_raw;

