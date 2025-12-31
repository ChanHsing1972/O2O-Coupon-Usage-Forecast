# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:37:21 2018

@author: FNo0
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
except ImportError:  # LightGBM 可选安装
    lgb = None

warnings.filterwarnings("ignore")  # 不显示警告


def prepare(dataset):
    """数据预处理

    1.时间处理(方便计算时间差):
        将Date_received列中int或float类型的元素转换成datetime类型,新增一列date_received存储;
        将Date列中int类型的元素转换为datetime类型,新增一列date存储;

    2.折扣处理:
        判断折扣率是“满减”(如10:1)还是“折扣率”(0.9);
        将“满减”折扣转换为“折扣率”形式(如10:1转换为0.9);
        得到“满减”折扣的最低消费(如折扣10:1的最低消费为10);
    3.距离处理:
        将空距离填充为-1(区别于距离0,1,2,3,4,5,6,7,8,9,10);
        判断是否为空距离;

    Args:
        dataset: DataFrame类型的数据集off_train和off_test,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'(off_test没有'Date'属性);

    Returns:
        预处理后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()
    # 折扣率处理
    data["is_manjian"] = data["Discount_rate"].map(
        lambda x: 1 if ":" in str(x) else 0
    )  # Discount_rate是否为满减
    data["discount_rate"] = data["Discount_rate"].map(
        lambda x: (
            float(x)
            if ":" not in str(x)
            else (float(str(x).split(":")[0]) - float(str(x).split(":")[1]))
            / float(str(x).split(":")[0])
        )
    )  # 满减全部转换为折扣率
    data["min_cost_of_manjian"] = data["Discount_rate"].map(
        lambda x: -1 if ":" not in str(x) else int(str(x).split(":")[0])
    )  # 满减的最低消费
    data["high_discount"] = data["discount_rate"].map(lambda x: 1 if x >= 0.85 else 0)
    data["discount_rate_sq"] = data["discount_rate"] * data["discount_rate"]
    data["discount_bucket"] = pd.cut(
        data["discount_rate"], bins=[-np.inf, 0.7, 0.8, 0.9, 1.0, np.inf], labels=False
    )
    # 距离处理
    data["Distance"] = pd.to_numeric(data["Distance"], errors="coerce")
    data["Distance"].fillna(11, inplace=True)  # 空距离填充为11（表示很远或未知）
    data["null_distance"] = data["Distance"].map(lambda x: 1 if x == 11 else 0)
    data["distance_bucket"] = pd.cut(
        data["Distance"], bins=[-np.inf, 0, 1, 2, 4, 9, np.inf], labels=False
    )
    # 填充Coupon_id
    data["Coupon_id"] = (
        pd.to_numeric(data["Coupon_id"], errors="coerce").fillna(0).astype(int)
    )

    # [新增] Discount_type 编码 (参考 wepe)
    discount_types = [
        "0.2",
        "0.5",
        "0.6",
        "0.7",
        "0.75",
        "0.8",
        "0.85",
        "0.9",
        "0.95",
        "30:20",
        "50:30",
        "10:5",
        "20:10",
        "100:50",
        "200:100",
        "50:20",
        "30:10",
        "150:50",
        "100:30",
        "20:5",
        "200:50",
        "5:1",
        "50:10",
        "100:20",
        "150:30",
        "30:5",
        "300:50",
        "200:30",
        "150:20",
        "10:1",
        "50:5",
        "100:10",
        "200:20",
        "300:30",
        "150:10",
        "300:20",
        "500:30",
        "20:1",
        "100:5",
        "200:10",
        "30:1",
        "150:5",
        "300:10",
        "200:5",
        "50:1",
        "100:1",
    ]
    data["discount_type"] = -1
    for k, v in enumerate(discount_types):
        data.loc[data["Discount_rate"] == v, "discount_type"] = k

    # 折扣-距离交互（弱非线性特征）
    try:
        data["discount_x_distance_bucket"] = data["discount_rate"] * data[
            "distance_bucket"
        ].astype(float)
    except Exception:
        data["discount_x_distance_bucket"] = 0.0
    # 时间处理
    data["date_received"] = pd.to_datetime(data["Date_received"], format="%Y%m%d")
    if "Date" in data.columns.tolist():  # off_train
        data["date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
        data["date_month"] = data["date"].dt.month
    # 返回
    return data


def get_label(dataset):
    """打标

    领取优惠券后15天内使用的样本标签为1,否则为0;

    Args:
        dataset: DataFrame类型的数据集off_train,包含属性'User_id','Merchant_id','Coupon_id','Discount_rate',
            'Distance','Date_received','Date'

    Returns:
        打标后的DataFrame类型的数据集.
    """
    # 源数据
    data = dataset.copy()
    # 打标:领券后15天内消费为1,否则为0
    data["label"] = list(
        map(
            lambda x, y: 1 if (x - y).total_seconds() / (60 * 60 * 24) <= 15 else 0,
            data["date"],
            data["date_received"],
        )
    )
    # 返回
    return data


def get_simple_feature(label_field):
    """简单的几个特征,作为初学示例

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data["Coupon_id"] = data["Coupon_id"].map(
        int
    )  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data["Date_received"] = data["Date_received"].map(
        int
    )  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data["cnt"] = 1  # 方便特征提取
    # 返回的特征数据集
    feature = data.copy()

    # 用户领券数
    keys = ["User_id"]  # 主键
    prefixs = "simple_" + "_".join(keys) + "_"  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(
        data, index=keys, values="cnt", aggfunc=len
    )  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "receive_cnt"})
        .reset_index()
    )  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how="left")  # 将id列与特征列左连

    # 用户领取特定优惠券数
    keys = ["User_id", "Coupon_id"]  # 主键
    prefixs = "simple_" + "_".join(keys) + "_"  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(
        data, index=keys, values="cnt", aggfunc=len
    )  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "receive_cnt"})
        .reset_index()
    )  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how="left")  # 将id列与特征列左连

    # 用户当天领券数
    keys = ["User_id", "Date_received"]  # 主键
    prefixs = "simple_" + "_".join(keys) + "_"  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(
        data, index=keys, values="cnt", aggfunc=len
    )  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "receive_cnt"})
        .reset_index()
    )  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how="left")  # 将id列与特征列左连

    # 用户当天领取特定优惠券数
    keys = ["User_id", "Coupon_id", "Date_received"]  # 主键
    prefixs = "simple_" + "_".join(keys) + "_"  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(
        data, index=keys, values="cnt", aggfunc=len
    )  # 以keys为键,'cnt'为值,使用len统计出现的次数
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "receive_cnt"})
        .reset_index()
    )  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how="left")  # 将id列与特征列左连

    # [新增] 用户当天领取特定商家的优惠券数
    keys = ["User_id", "Merchant_id", "Date_received"]
    prefixs = "simple_" + "_".join(keys) + "_"
    pivot = pd.pivot_table(data, index=keys, values="cnt", aggfunc=len)
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "receive_cnt"})
        .reset_index()
    )
    feature = pd.merge(feature, pivot, on=keys, how="left")

    # [新增] 当天全网领券数 (反映节日/大促)
    keys = ["Date_received"]
    prefixs = "simple_" + "_".join(keys) + "_"
    pivot = pd.pivot_table(data, index=keys, values="cnt", aggfunc=len)
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "receive_cnt"})
        .reset_index()
    )
    feature = pd.merge(feature, pivot, on=keys, how="left")

    # 用户是否在同一天重复领取了特定优惠券
    keys = ["User_id", "Coupon_id", "Date_received"]  # 主键
    prefixs = "simple_" + "_".join(keys) + "_"  # 特征名前缀,由label_field和主键组成
    pivot = pd.pivot_table(
        data, index=keys, values="cnt", aggfunc=lambda x: 1 if len(x) > 1 else 0
    )  # 以keys为键,'cnt'为值,判断领取次数是否大于1
    pivot = (
        pd.DataFrame(pivot)
        .rename(columns={"cnt": prefixs + "repeat_receive"})
        .reset_index()
    )  # pivot_table后keys会成为index,统计出的特征列会以values即'cnt'命名,将其改名为特征名前缀+特征意义,并将index还原
    feature = pd.merge(feature, pivot, on=keys, how="left")  # 将id列与特征列左连

    # 删除辅助提特征的'cnt'
    feature.drop(["cnt"], axis=1, inplace=True)

    # 返回
    return feature


def get_week_feature(label_field):
    """根据Date_received得到的一些日期特征

    根据date_received列得到领券日是周几,新增一列week存储,并将其one-hot离散为week_0,week_1,week_2,week_3,week_4,week_5,week_6;
    根据week列得到领券日是否为休息日,新增一列is_weekend存储;

    Args:

    Returns:

    """
    # 源数据
    data = label_field.copy()
    data["Coupon_id"] = data["Coupon_id"].map(
        int
    )  # 将Coupon_id列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    data["Date_received"] = data["Date_received"].map(
        int
    )  # 将Date_received列中float类型的元素转换为int类型,因为列中存在np.nan即空值会让整列的元素变为float
    # 返回的特征数据集
    feature = data.copy()
    feature["week"] = feature["date_received"].map(lambda x: x.weekday())  # 星期几
    feature["is_weekend"] = feature["week"].map(
        lambda x: 1 if x == 5 or x == 6 else 0
    )  # 判断领券日是否为休息日
    feature["receive_month"] = feature["date_received"].dt.month
    feature["receive_day"] = feature["date_received"].dt.day
    feature["receive_weekofmonth"] = (feature["receive_day"] - 1) // 7
    feature["is_month_start"] = feature["date_received"].dt.is_month_start.astype(int)
    feature["is_month_end"] = feature["date_received"].dt.is_month_end.astype(int)
    # 周期特征（避免纯整数编码的序数偏差）
    feature["week_sin"] = np.sin(2 * np.pi * feature["week"] / 7)
    feature["week_cos"] = np.cos(2 * np.pi * feature["week"] / 7)
    feature = pd.concat(
        [feature, pd.get_dummies(feature["week"], prefix="week")], axis=1
    )  # one-hot离散星期几
    feature.index = range(len(feature))  # 重置index
    # 返回
    return feature


def safe_divide(numer, denom):
    numer = numer.astype(float)
    denom = denom.replace(0, np.nan)
    return (numer / denom).fillna(0)


def bayes_smooth_rate(pos_cnt, total_cnt, prior, m=20.0):
    """贝叶斯平滑： (pos + prior*m) / (total + m)

    - prior: 全局转化率先验（0~1）
    - m: 先验强度，越大越“向全局靠拢”
    """
    pos = pd.to_numeric(pos_cnt, errors="coerce").fillna(0).astype(float)
    tot = pd.to_numeric(total_cnt, errors="coerce").fillna(0).astype(float)
    prior = float(prior) if prior is not None else 0.0
    m = float(m)
    return (pos + prior * m) / (tot + m)


def group_rank_pct(df, group_col, value_col):
    """组内分位rank，返回[0,1]，缺失时填0.5。"""
    s = df.groupby(group_col)[value_col].rank(method="average", pct=True)
    return s.fillna(0.5).astype(float)


def group_zscore(df, group_col, value_col):
    """组内z-score标准化，缺失/方差0时返回0。"""
    g = df.groupby(group_col)[value_col]
    mean = g.transform("mean")
    std = g.transform("std").replace(0, np.nan)
    z = (df[value_col] - mean) / std
    return z.fillna(0.0).astype(float)


def build_history_features(history_field):
    """从历史区间提取用户/商家/优惠券/交互统计特征"""
    if history_field.empty:
        return {}, pd.DataFrame()

    data = history_field.copy()
    data["used"] = data["Date"].notnull().astype(int)

    # 区分券记录和普通消费记录
    # 券记录: Coupon_id != 0
    # 普通消费: Coupon_id == 0
    coupon_data = data[data["Coupon_id"] != 0].copy()
    regular_data = data[data["Coupon_id"] == 0].copy()

    # 全局先验 (仅基于券)
    global_prior = float(coupon_data["used"].mean()) if len(coupon_data) else 0.0

    # ==========================================
    # 1. 用户维度 (User)
    # ==========================================
    # 基于券数据的统计
    u_grp = coupon_data.groupby("User_id")
    user_feat = pd.DataFrame({"User_id": u_grp.size().index})
    user_feat["user_receive_cnt"] = u_grp.size().values
    user_feat["user_used_cnt"] = u_grp["used"].sum().values
    user_feat["user_merchant_count"] = u_grp["Merchant_id"].nunique().values
    user_feat["user_coupon_count"] = u_grp["Coupon_id"].nunique().values
    user_feat["user_distance_mean"] = (
        u_grp["Distance"].apply(lambda x: x.replace(11, np.nan).mean()).values
    )
    user_feat["user_distance_min"] = (
        u_grp["Distance"].apply(lambda x: x.replace(11, np.nan).min()).values
    )
    user_feat["user_discount_mean"] = u_grp["discount_rate"].mean().values

    # 基于普通消费的统计
    r_grp = regular_data.groupby("User_id")
    r_feat = pd.DataFrame({"User_id": r_grp.size().index})
    r_feat["user_regular_cnt"] = r_grp.size().values

    # 合并
    user_feat = pd.merge(user_feat, r_feat, on="User_id", how="outer")

    # Fill counts
    cnt_cols = [
        "user_receive_cnt",
        "user_used_cnt",
        "user_regular_cnt",
        "user_merchant_count",
        "user_coupon_count",
    ]
    user_feat[cnt_cols] = user_feat[cnt_cols].fillna(0)

    # Calculate rates
    user_feat["user_used_rate"] = bayes_smooth_rate(
        user_feat["user_used_cnt"], user_feat["user_receive_cnt"], global_prior, m=10
    )
    user_feat["user_total_consume_cnt"] = (
        user_feat["user_used_cnt"] + user_feat["user_regular_cnt"]
    )
    user_feat["user_coupon_consume_rate"] = safe_divide(
        user_feat["user_used_cnt"], user_feat["user_total_consume_cnt"]
    )

    # 核销过的统计 (User Used Stats)
    user_used_data = coupon_data[coupon_data["used"] == 1]
    if not user_used_data.empty:
        uu_grp = user_used_data.groupby("User_id")
        uu_feat = pd.DataFrame({"User_id": uu_grp.size().index})
        uu_feat["user_mean_discount_used"] = uu_grp["discount_rate"].mean().values
        uu_feat["user_min_discount_used"] = uu_grp["discount_rate"].min().values
        uu_feat["user_max_discount_used"] = uu_grp["discount_rate"].max().values

        uu_feat["user_mean_distance_used"] = (
            uu_grp["Distance"].apply(lambda x: x.replace(11, np.nan).mean()).values
        )
        uu_feat["user_min_distance_used"] = (
            uu_grp["Distance"].apply(lambda x: x.replace(11, np.nan).min()).values
        )
        uu_feat["user_max_distance_used"] = (
            uu_grp["Distance"].apply(lambda x: x.replace(11, np.nan).max()).values
        )

        uu_feat["user_distinct_coupon_used_count"] = (
            uu_grp["Coupon_id"].nunique().values
        )

        user_feat = user_feat.merge(uu_feat, on="User_id", how="left")

        # Distinct Coupon Used Rate
        user_feat["user_distinct_coupon_used_rate"] = safe_divide(
            user_feat["user_distinct_coupon_used_count"], user_feat["user_coupon_count"]
        )

    # [新增] 时间间隔特征 (Gap Features)
    def calc_gap(dates):
        if len(dates) < 2:
            return -1
        dates = sorted(dates)
        return (dates[-1] - dates[0]).days / (len(dates) - 1)

    consume_data = data[data["date"].notnull()][["User_id", "date", "Coupon_id"]]

    # 普通消费间隔
    reg_consume = consume_data[consume_data["Coupon_id"] == 0]
    if not reg_consume.empty:
        reg_gap = reg_consume.groupby("User_id")["date"].apply(calc_gap).reset_index()
        reg_gap.columns = ["User_id", "user_regular_gap"]
        user_feat = user_feat.merge(reg_gap, on="User_id", how="left")

    # 优惠券消费间隔
    coupon_consume = consume_data[consume_data["Coupon_id"] != 0]
    if not coupon_consume.empty:
        coupon_gap = (
            coupon_consume.groupby("User_id")["date"].apply(calc_gap).reset_index()
        )
        coupon_gap.columns = ["User_id", "user_coupon_gap"]
        user_feat = user_feat.merge(coupon_gap, on="User_id", how="left")

    # 领券到核销的平均间隔
    coupon_consume_full = data[
        (data["Coupon_id"] != 0) & (data["date"].notnull())
    ].copy()
    if not coupon_consume_full.empty:
        coupon_consume_full["gap"] = (
            coupon_consume_full["date"] - coupon_consume_full["date_received"]
        ).dt.days
        avg_gap = coupon_consume_full.groupby("User_id")["gap"].mean().reset_index()
        avg_gap.columns = ["User_id", "user_avg_consume_gap"]
        user_feat = user_feat.merge(avg_gap, on="User_id", how="left")

    user_feat.fillna(-1, inplace=True)

    # ==========================================
    # 2. 商家维度 (Merchant)
    # ==========================================
    m_grp = coupon_data.groupby("Merchant_id")
    merchant_feat = pd.DataFrame({"Merchant_id": m_grp.size().index})
    merchant_feat["merchant_receive_cnt"] = m_grp.size().values
    merchant_feat["merchant_used_cnt"] = m_grp["used"].sum().values
    merchant_feat["merchant_distance_mean"] = (
        m_grp["Distance"].apply(lambda x: x.replace(11, np.nan).mean()).values
    )
    merchant_feat["merchant_discount_mean"] = m_grp["discount_rate"].mean().values

    mr_grp = regular_data.groupby("Merchant_id")
    mr_feat = pd.DataFrame({"Merchant_id": mr_grp.size().index})
    mr_feat["merchant_regular_cnt"] = mr_grp.size().values

    merchant_feat = pd.merge(merchant_feat, mr_feat, on="Merchant_id", how="outer")

    cnt_cols_m = ["merchant_receive_cnt", "merchant_used_cnt", "merchant_regular_cnt"]
    merchant_feat[cnt_cols_m] = merchant_feat[cnt_cols_m].fillna(0)

    merchant_feat["merchant_used_rate"] = bayes_smooth_rate(
        merchant_feat["merchant_used_cnt"],
        merchant_feat["merchant_receive_cnt"],
        global_prior,
        m=10,
    )
    merchant_feat["merchant_total_consume_cnt"] = (
        merchant_feat["merchant_used_cnt"] + merchant_feat["merchant_regular_cnt"]
    )
    merchant_feat["merchant_coupon_consume_rate"] = safe_divide(
        merchant_feat["merchant_used_cnt"], merchant_feat["merchant_total_consume_cnt"]
    )

    # [新增] 商家用户留存/活跃特征
    # merchant_feat["merchant_distinct_user_received_count"] = m_grp["User_id"].nunique().values # Error: Length mismatch
    temp_nunique = m_grp["User_id"].nunique().reset_index()
    temp_nunique.columns = ["Merchant_id", "merchant_distinct_user_received_count"]
    merchant_feat = merchant_feat.merge(temp_nunique, on="Merchant_id", how="left")

    m_used_data = coupon_data[(coupon_data["used"] == 1)]
    if not m_used_data.empty:
        # 商家被多少不同用户核销
        mu_distinct = (
            m_used_data.groupby("Merchant_id")["User_id"].nunique().reset_index()
        )
        mu_distinct.columns = ["Merchant_id", "merchant_distinct_user_used_count"]
        merchant_feat = merchant_feat.merge(mu_distinct, on="Merchant_id", how="left")

        # 商家距离特征
        m_used_dist = m_used_data[m_used_data["Distance"] != 11]
        if not m_used_dist.empty:
            mu_dist = (
                m_used_dist.groupby("Merchant_id")["Distance"].mean().reset_index()
            )
            mu_dist.columns = ["Merchant_id", "merchant_redeem_dist_mean"]
            merchant_feat = merchant_feat.merge(mu_dist, on="Merchant_id", how="left")

    merchant_feat.fillna(-1, inplace=True)

    # 商家用户留存率 (核销用户数 / 领券用户数)
    merchant_feat["merchant_user_retention_rate"] = safe_divide(
        merchant_feat["merchant_distinct_user_used_count"],
        merchant_feat["merchant_distinct_user_received_count"],
    )
    # 商家平均每个用户核销多少张
    merchant_feat["merchant_avg_user_usage"] = safe_divide(
        merchant_feat["merchant_used_cnt"],
        merchant_feat["merchant_distinct_user_received_count"],
    )

    # ==========================================
    # 3. 优惠券维度 (Coupon)
    # ==========================================
    c_grp = coupon_data.groupby("Coupon_id")
    coupon_feat = pd.DataFrame({"Coupon_id": c_grp.size().index})
    coupon_feat["coupon_receive_cnt"] = c_grp.size().values
    coupon_feat["coupon_used_cnt"] = c_grp["used"].sum().values
    coupon_feat["coupon_used_rate"] = bayes_smooth_rate(
        coupon_feat["coupon_used_cnt"],
        coupon_feat["coupon_receive_cnt"],
        global_prior,
        m=10,
    )
    coupon_feat["coupon_discount"] = c_grp["discount_rate"].mean().values

    # [新增] 优惠券平均核销时间
    c_used_data = coupon_data[coupon_data["used"] == 1].copy()
    if not c_used_data.empty:
        c_used_data["gap"] = (
            c_used_data["date"] - c_used_data["date_received"]
        ).dt.days
        c_gap = c_used_data.groupby("Coupon_id")["gap"].mean().reset_index()
        c_gap.columns = ["Coupon_id", "coupon_avg_redeem_gap"]
        coupon_feat = coupon_feat.merge(c_gap, on="Coupon_id", how="left")

    coupon_feat.fillna(-1, inplace=True)

    # ==========================================
    # 4. 交互维度 (UM, UC, MC)
    # ==========================================
    # UM
    um_grp = coupon_data.groupby(["User_id", "Merchant_id"])
    um_feat = pd.DataFrame(
        {
            "User_id": [u for u, _ in um_grp.size().index],
            "Merchant_id": [m for _, m in um_grp.size().index],
        }
    )
    um_feat["um_receive_cnt"] = um_grp.size().values
    um_feat["um_used_cnt"] = um_grp["used"].sum().values

    umr_grp = regular_data.groupby(["User_id", "Merchant_id"])
    umr_feat = pd.DataFrame(
        {
            "User_id": [u for u, _ in umr_grp.size().index],
            "Merchant_id": [m for _, m in umr_grp.size().index],
        }
    )
    umr_feat["um_regular_cnt"] = umr_grp.size().values

    um_feat = pd.merge(um_feat, umr_feat, on=["User_id", "Merchant_id"], how="outer")

    cnt_cols_um = ["um_receive_cnt", "um_used_cnt", "um_regular_cnt"]
    um_feat[cnt_cols_um] = um_feat[cnt_cols_um].fillna(0)

    um_feat["um_used_rate"] = bayes_smooth_rate(
        um_feat["um_used_cnt"], um_feat["um_receive_cnt"], global_prior, m=10
    )
    um_feat["um_total_consume_cnt"] = um_feat["um_used_cnt"] + um_feat["um_regular_cnt"]
    um_feat["um_coupon_consume_rate"] = safe_divide(
        um_feat["um_used_cnt"], um_feat["um_total_consume_cnt"]
    )

    # UM Ratios
    um_feat = um_feat.merge(
        user_feat[["User_id", "user_used_cnt"]], on="User_id", how="left"
    )
    um_feat = um_feat.merge(
        merchant_feat[["Merchant_id", "merchant_used_cnt"]],
        on="Merchant_id",
        how="left",
    )
    um_feat["um_buy_rate"] = safe_divide(
        um_feat["um_used_cnt"], um_feat["user_used_cnt"]
    )
    um_feat["um_merchant_rate"] = safe_divide(
        um_feat["um_used_cnt"], um_feat["merchant_used_cnt"]
    )
    um_feat["um_visit_rate"] = safe_divide(
        um_feat["um_used_cnt"], um_feat["um_receive_cnt"]
    )
    um_feat.drop(["user_used_cnt", "merchant_used_cnt"], axis=1, inplace=True)

    # UM Ranks
    um_feat["um_user_rank_in_merchant"] = um_feat.groupby("Merchant_id")[
        "um_total_consume_cnt"
    ].rank(ascending=False, method="min")
    um_feat["um_merchant_rank_in_user"] = um_feat.groupby("User_id")[
        "um_total_consume_cnt"
    ].rank(ascending=False, method="min")

    um_feat.fillna(-1, inplace=True)

    # UC
    uc_grp = coupon_data.groupby(["User_id", "Coupon_id"])
    uc_feat = pd.DataFrame(
        {
            "User_id": [u for u, _ in uc_grp.size().index],
            "Coupon_id": [c for _, c in uc_grp.size().index],
        }
    )
    uc_feat["uc_receive_cnt"] = uc_grp.size().values
    uc_feat["uc_used_cnt"] = uc_grp["used"].sum().values
    uc_feat["uc_used_rate"] = bayes_smooth_rate(
        uc_feat["uc_used_cnt"], uc_feat["uc_receive_cnt"], global_prior, m=10
    )
    uc_feat.fillna(-1, inplace=True)

    # MC
    mc_grp = coupon_data.groupby(["Merchant_id", "Coupon_id"])
    mc_feat = pd.DataFrame(
        {
            "Merchant_id": [m for m, _ in mc_grp.size().index],
            "Coupon_id": [c for _, c in mc_grp.size().index],
        }
    )
    mc_feat["mc_receive_cnt"] = mc_grp.size().values
    mc_feat["mc_used_cnt"] = mc_grp["used"].sum().values
    mc_feat["mc_used_rate"] = bayes_smooth_rate(
        mc_feat["mc_used_cnt"], mc_feat["mc_receive_cnt"], global_prior, m=10
    )
    mc_feat.fillna(-1, inplace=True)

    return {
        "user": user_feat,
        "merchant": merchant_feat,
        "coupon": coupon_feat,
        "um": um_feat,
        "uc": uc_feat,
        "mc": mc_feat,
    }, data[["User_id", "Coupon_id", "Date_received", "Date"]]


def build_online_features(online_df):
    """从线上行为构造用户/商家/优惠券/交互统计特征"""
    if online_df is None or online_df.empty:
        return {}

    data = online_df.copy()
    data["User_id"] = pd.to_numeric(data["User_id"], errors="coerce")
    data["Merchant_id"] = pd.to_numeric(data["Merchant_id"], errors="coerce")
    data["Coupon_id"] = pd.to_numeric(data["Coupon_id"], errors="coerce")
    data.dropna(subset=["User_id", "Merchant_id", "Coupon_id"], inplace=True)
    data = data.astype({"User_id": int, "Merchant_id": int, "Coupon_id": int})
    # Action: 0点击 1购买 2领券
    data["is_click"] = (data["Action"] == 0).astype(int)
    data["is_buy"] = (data["Action"] == 1).astype(int)
    data["is_receive"] = (data["Action"] == 2).astype(int)

    user_grp = data.groupby("User_id")
    user_feat = pd.DataFrame({"User_id": user_grp.size().index})
    user_feat["user_online_click"] = user_grp["is_click"].sum().values
    user_feat["user_online_buy"] = user_grp["is_buy"].sum().values
    user_feat["user_online_receive"] = user_grp["is_receive"].sum().values
    user_feat["user_online_action_count"] = user_grp.size().values
    user_feat["user_online_distinct_merchant_count"] = (
        user_grp["Merchant_id"].nunique().values
    )
    user_feat["user_online_buy_rate"] = safe_divide(
        user_feat["user_online_buy"],
        user_feat["user_online_click"] + user_feat["user_online_receive"],
    )

    merchant_grp = data.groupby("Merchant_id")
    merchant_feat = pd.DataFrame({"Merchant_id": merchant_grp.size().index})
    merchant_feat["merchant_online_click"] = merchant_grp["is_click"].sum().values
    merchant_feat["merchant_online_buy"] = merchant_grp["is_buy"].sum().values
    merchant_feat["merchant_online_receive"] = merchant_grp["is_receive"].sum().values
    merchant_feat["merchant_online_action_count"] = merchant_grp.size().values
    merchant_feat["merchant_online_distinct_user_count"] = (
        merchant_grp["User_id"].nunique().values
    )
    merchant_feat["merchant_online_buy_rate"] = safe_divide(
        merchant_feat["merchant_online_buy"],
        merchant_feat["merchant_online_click"]
        + merchant_feat["merchant_online_receive"],
    )

    coupon_grp = data.groupby("Coupon_id")
    coupon_feat = pd.DataFrame({"Coupon_id": coupon_grp.size().index})
    coupon_feat["coupon_online_receive"] = coupon_grp["is_receive"].sum().values
    coupon_feat["coupon_online_buy"] = coupon_grp["is_buy"].sum().values
    coupon_feat["coupon_online_buy_rate"] = safe_divide(
        coupon_feat["coupon_online_buy"],
        coupon_feat["coupon_online_receive"],
    )

    um_grp = data.groupby(["User_id", "Merchant_id"])
    um_feat = pd.DataFrame(
        {
            "User_id": [u for u, _ in um_grp.size().index],
            "Merchant_id": [m for _, m in um_grp.size().index],
        }
    )
    um_feat["um_online_click"] = um_grp["is_click"].sum().values
    um_feat["um_online_buy"] = um_grp["is_buy"].sum().values
    um_feat["um_online_receive"] = um_grp["is_receive"].sum().values
    um_feat["um_online_buy_rate"] = safe_divide(
        um_feat["um_online_buy"],
        um_feat["um_online_click"] + um_feat["um_online_receive"],
    )

    return {
        "user": user_feat,
        "merchant": merchant_feat,
        "coupon": coupon_feat,
        "um": um_feat,
    }


def build_coupon_weekday_rates(history_full):
    """基于领券星期的券使用率（历史区间）

    以 history_full 的领券日 weekday 与是否使用（Date 非空）统计券在不同星期的转化率。
    """
    if history_full is None or history_full.empty:
        return pd.DataFrame(columns=["Coupon_id", "week", "coupon_week_used_rate"])

    df = history_full.copy()
    df["week"] = df["date_received"].dt.weekday
    df["used"] = df["Date"].notnull().astype(int)
    grp = df.groupby(["Coupon_id", "week"])  # 按券+星期分组
    out = grp["used"].agg(["sum", "count"]).reset_index()
    out.rename(columns={"sum": "used_cnt", "count": "receive_cnt"}, inplace=True)
    global_prior = float(df["used"].mean()) if len(df) else 0.0
    out["coupon_week_used_rate"] = bayes_smooth_rate(
        out["used_cnt"], out["receive_cnt"], global_prior, m=20
    )
    return out[["Coupon_id", "week", "coupon_week_used_rate"]]


def add_user_recency(label_field):
    """计算用户领券的前/后一次间隔天数"""
    tmp = label_field[["User_id", "date_received", "row_id"]].copy()
    tmp.sort_values(["User_id", "date_received", "row_id"], inplace=True)
    tmp["prev_date"] = tmp.groupby("User_id")["date_received"].shift(1)
    tmp["next_date"] = tmp.groupby("User_id")["date_received"].shift(-1)
    tmp["user_gap_since_last"] = (tmp["date_received"] - tmp["prev_date"]).dt.days
    tmp["user_gap_until_next"] = (tmp["next_date"] - tmp["date_received"]).dt.days
    tmp[["user_gap_since_last", "user_gap_until_next"]] = tmp[
        ["user_gap_since_last", "user_gap_until_next"]
    ].fillna(9999)
    tmp = tmp[["User_id", "row_id", "user_gap_since_last", "user_gap_until_next"]]
    return tmp


def add_um_recency(label_field):
    """计算同一用户-商家领券的前/后一次间隔天数"""
    tmp = label_field[["User_id", "Merchant_id", "date_received", "row_id"]].copy()
    tmp.sort_values(["User_id", "Merchant_id", "date_received", "row_id"], inplace=True)
    tmp["prev_date"] = tmp.groupby(["User_id", "Merchant_id"])["date_received"].shift(1)
    tmp["next_date"] = tmp.groupby(["User_id", "Merchant_id"])["date_received"].shift(
        -1
    )
    tmp["um_gap_since_last"] = (tmp["date_received"] - tmp["prev_date"]).dt.days
    tmp["um_gap_until_next"] = (tmp["next_date"] - tmp["date_received"]).dt.days
    tmp[["um_gap_since_last", "um_gap_until_next"]] = tmp[
        ["um_gap_since_last", "um_gap_until_next"]
    ].fillna(9999)
    tmp = tmp[["row_id", "um_gap_since_last", "um_gap_until_next"]]
    return tmp


def add_uc_recency(label_field):
    """计算同一用户-券的前/后一次领取间隔天数"""
    tmp = label_field[["User_id", "Coupon_id", "date_received", "row_id"]].copy()
    tmp.sort_values(["User_id", "Coupon_id", "date_received", "row_id"], inplace=True)
    tmp["prev_date"] = tmp.groupby(["User_id", "Coupon_id"])["date_received"].shift(1)
    tmp["next_date"] = tmp.groupby(["User_id", "Coupon_id"])["date_received"].shift(-1)
    tmp["uc_gap_since_last"] = (tmp["date_received"] - tmp["prev_date"]).dt.days
    tmp["uc_gap_until_next"] = (tmp["next_date"] - tmp["date_received"]).dt.days
    tmp[["uc_gap_since_last", "uc_gap_until_next"]] = tmp[
        ["uc_gap_since_last", "uc_gap_until_next"]
    ].fillna(9999)
    tmp = tmp[["row_id", "uc_gap_since_last", "uc_gap_until_next"]]
    return tmp


def add_sequence_features(label_field):
    """领取顺序特征：用户/用户-券维度的时间排序"""
    tmp = label_field[["row_id", "User_id", "Coupon_id", "date_received"]].copy()
    tmp.sort_values(["User_id", "date_received", "row_id"], inplace=True)
    tmp["user_receive_rank"] = tmp.groupby("User_id")["date_received"].rank(
        method="first"
    )
    tmp["user_receive_rev_rank"] = tmp.groupby("User_id")["date_received"].rank(
        method="first", ascending=False
    )

    tmp.sort_values(["User_id", "Coupon_id", "date_received", "row_id"], inplace=True)
    tmp["uc_receive_rank"] = tmp.groupby(["User_id", "Coupon_id"])[
        "date_received"
    ].rank(method="first")
    tmp["uc_receive_rev_rank"] = tmp.groupby(["User_id", "Coupon_id"])[
        "date_received"
    ].rank(method="first", ascending=False)

    return tmp[
        [
            "row_id",
            "user_receive_rank",
            "user_receive_rev_rank",
            "uc_receive_rank",
            "uc_receive_rev_rank",
        ]
    ]


def evaluate_coupon_auc(df, prob_col="prob"):
    """按coupon_id分组求AUC的均值"""
    auc_list = []
    for _, group in df.groupby("Coupon_id"):
        if group["label"].nunique() < 2:
            continue
        auc_list.append(roc_auc_score(group["label"], group[prob_col]))
    return float(np.mean(auc_list)) if auc_list else 0.0


def dedup_result(result_df):
    """去重同一(user,coupon,date_received)取最大prob"""
    result_df = result_df.sort_values(["prob"], ascending=False)
    result_df = result_df.drop_duplicates(
        subset=["User_id", "Coupon_id", "Date_received"], keep="first"
    )
    result_df.index = range(len(result_df))
    return result_df


def _recent_counts(base_group, hist_dates, used_dates, window_days):
    """Compute rolling counts within past window_days for a subset (numpy arrays)."""
    if hist_dates.size == 0:
        return np.zeros(len(base_group), dtype=int), np.zeros(
            len(base_group), dtype=int
        )
    # convert datetimes to day integers
    hist_int = hist_dates.astype("datetime64[D]").astype(np.int64)
    used_int = (
        used_dates.astype("datetime64[D]").astype(np.int64)
        if used_dates.size
        else hist_int[:0]
    )
    label_int = (
        base_group["date_received"].to_numpy().astype("datetime64[D]").astype(np.int64)
    )
    low = label_int - window_days
    # counts where low < hist_date < label_date
    left = np.searchsorted(hist_int, low, side="right")
    right = np.searchsorted(hist_int, label_int, side="left")
    total = right - left
    if used_int.size:
        l_used = np.searchsorted(used_int, low, side="right")
        r_used = np.searchsorted(used_int, label_int, side="left")
        used = r_used - l_used
    else:
        used = np.zeros(len(label_int), dtype=int)
    return total, used


def build_recent_window_features(base, history_full, windows=(7, 15, 30)):
    """Rolling-window counts for recent behavior (multi-window)."""
    # 全局先验（历史区间整体转化率），用于近期窗口rate平滑
    global_prior = (
        float(history_full["Date"].notnull().mean())
        if history_full is not None and len(history_full)
        else 0.0
    )
    m_smooth = 5.0
    key_defs = [
        (["User_id"], "user"),
        (["Merchant_id"], "merchant"),
        (["Coupon_id"], "coupon"),
        (["User_id", "Merchant_id"], "um"),
        (["User_id", "Coupon_id"], "uc"),
        (["Merchant_id", "Coupon_id"], "mc"),
    ]
    out = pd.DataFrame({"row_id": base["row_id"]})
    for keys, prefix in key_defs:
        hist_grp = history_full.groupby(keys)
        base_sub = base[keys + ["row_id", "date_received"]].copy()
        rows = []
        for k, sub_df in base_sub.groupby(keys):
            if not isinstance(k, tuple):
                k = (k,)
            if k in hist_grp.groups:
                hist_df = hist_grp.get_group(k)
                # [Fix] Filter NaT and sort for searchsorted
                hist_df = hist_df[hist_df["date_received"].notnull()].sort_values(
                    "date_received"
                )
                hist_dates = hist_df["date_received"].to_numpy()
                used_dates = hist_df[hist_df["Date"].notnull()][
                    "date_received"
                ].to_numpy()
            else:
                hist_dates = np.array([], dtype="datetime64[ns]")
                used_dates = np.array([], dtype="datetime64[ns]")

            rec = {"row_id": sub_df["row_id"].to_numpy()}
            for w in windows:
                total, used = _recent_counts(sub_df, hist_dates, used_dates, w)
                rec[f"{prefix}_recent_receive_{w}"] = total
                rec[f"{prefix}_recent_used_{w}"] = used
                # 转化率特征：贝叶斯平滑，减少稀疏噪声
                rec[f"{prefix}_recent_rate_{w}"] = (
                    used.astype(float) + global_prior * m_smooth
                ) / (total.astype(float) + m_smooth)
            rows.append(pd.DataFrame(rec))
        if rows:
            feat_block = pd.concat(rows, axis=0)
            out = out.merge(feat_block, on="row_id", how="left")
    return out


def blend_results(df_a, df_b, w_a=0.6, w_b=0.4):
    """按权重融合两个提交结果，权重和需为1"""
    merged = df_a.merge(
        df_b,
        on=["User_id", "Coupon_id", "Date_received"],
        how="inner",
        suffixes=("_a", "_b"),
    )
    merged["prob"] = merged["prob_a"] * w_a + merged["prob_b"] * w_b
    return merged[["User_id", "Coupon_id", "Date_received", "prob"]]


def get_dataset(history_field, middle_field, label_field, online_feats=None):
    """构造数据集

    Args:

    Returns:

    """
    base = label_field.copy()
    base["row_id"] = base.index

    # 特征工程
    week_feat = get_week_feature(base)
    simple_feat = get_simple_feature(base)
    recency_feat = add_user_recency(base)
    um_recency_feat = add_um_recency(base)
    uc_recency_feat = add_uc_recency(base)
    seq_feat = add_sequence_features(base)
    history_full = pd.concat([history_field, middle_field], axis=0)
    history_feats, _ = build_history_features(history_full)
    recent_feats = build_recent_window_features(base, history_full, windows=(7, 15, 30))
    # 券-星期使用率特征
    coupon_week_rates = build_coupon_weekday_rates(history_full)

    # 构造数据集
    share_characters = list(
        set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist())
    )
    dataset = pd.concat([week_feat, simple_feat.drop(share_characters, axis=1)], axis=1)
    # 关联券-星期使用率
    dataset = dataset.merge(
        coupon_week_rates,
        on=["Coupon_id", "week"],
        how="left",
    )
    dataset = dataset.merge(recency_feat, on=["row_id", "User_id"], how="left")
    dataset = dataset.merge(um_recency_feat, on="row_id", how="left")
    dataset = dataset.merge(uc_recency_feat, on="row_id", how="left")
    dataset = dataset.merge(seq_feat, on="row_id", how="left")
    dataset = dataset.merge(recent_feats, on="row_id", how="left")

    # 关联历史统计
    if history_feats:
        if "user" in history_feats:
            dataset = dataset.merge(history_feats["user"], on="User_id", how="left")

        if "merchant" in history_feats:
            dataset = dataset.merge(
                history_feats["merchant"], on="Merchant_id", how="left"
            )
        if "coupon" in history_feats:
            dataset = dataset.merge(history_feats["coupon"], on="Coupon_id", how="left")
        if "um" in history_feats:
            dataset = dataset.merge(
                history_feats["um"], on=["User_id", "Merchant_id"], how="left"
            )
        if "uc" in history_feats:
            dataset = dataset.merge(
                history_feats["uc"], on=["User_id", "Coupon_id"], how="left"
            )
        if "mc" in history_feats:
            dataset = dataset.merge(
                history_feats["mc"], on=["Merchant_id", "Coupon_id"], how="left"
            )

    # 关联线上统计
    if online_feats:
        if "user" in online_feats:
            dataset = dataset.merge(online_feats["user"], on="User_id", how="left")
        if "merchant" in online_feats:
            dataset = dataset.merge(
                online_feats["merchant"], on="Merchant_id", how="left"
            )
        if "coupon" in online_feats:
            dataset = dataset.merge(online_feats["coupon"], on="Coupon_id", how="left")
        if "um" in online_feats:
            dataset = dataset.merge(
                online_feats["um"], on=["User_id", "Merchant_id"], how="left"
            )

    # 删除无用属性并将label置于最后一列
    drop_cols = ["Merchant_id", "Discount_rate", "date_received", "date_month"]
    if "Date" in dataset.columns.tolist():  # 表示训练集和验证集
        drop_cols.append("Date")
        drop_cols.append("date")
        label = dataset["label"].tolist()
        dataset.drop(["label"], axis=1, inplace=True)
        dataset["label"] = label

    # 某些切片缺少部分时间列，忽略缺失安全删除
    dataset.drop(drop_cols, axis=1, inplace=True, errors="ignore")

    # 修正数据类型
    dataset["User_id"] = dataset["User_id"].map(int)
    dataset["Coupon_id"] = dataset["Coupon_id"].map(int)
    dataset["Date_received"] = dataset["Date_received"].map(int)
    dataset["Distance"] = dataset["Distance"].map(int)
    if "label" in dataset.columns.tolist():
        dataset["label"] = dataset["label"].map(int)

    # 填充缺失统计特征
    dataset.fillna(0, inplace=True)

    # 去重
    dataset.drop(["row_id"], axis=1, inplace=True)
    dataset.drop_duplicates(keep="first", inplace=True)
    dataset.index = range(len(dataset))
    # 返回
    return dataset


def model_xgb(train, valid=None, test=None, n_models=1, seeds=None):
    """XGBoost模型，支持验证/测试预测，用于融合"""
    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.01,
        "max_depth": 5,
        "min_child_weight": 1.1,
        "gamma": 0.1,
        "lambda": 10,
        "colsample_bylevel": 0.7,
        "colsample_bytree": 0.7,
        "subsample": 0.7,
        "scale_pos_weight": 1,
        "verbosity": 1,
    }

    feature_cols = [
        c
        for c in train.columns
        if c not in ["User_id", "Coupon_id", "Date_received", "label"]
    ]
    print(
        f"XGBoost 使用特征数: {len(feature_cols)}, 训练样本: {len(train)}, 验证样本: {len(valid) if valid is not None else 0}, 测试样本: {len(test) if test is not None else 0}",
        flush=True,
    )

    dtrain = xgb.DMatrix(train[feature_cols], label=train["label"])
    watchlist = [(dtrain, "train")]
    dvalid = None
    if valid is not None:
        dvalid = xgb.DMatrix(valid[feature_cols], label=valid["label"])
        watchlist.append((dvalid, "valid"))

    # 多种子集成（不做扰动，避免线上不一致）
    if seeds is None or n_models <= 1:
        models = [
            xgb.train(
                params,
                dtrain,
                num_boost_round=3000,
                evals=watchlist,
                early_stopping_rounds=200 if dvalid is not None else None,
                verbose_eval=100,
            )
        ]
    else:
        models = []
        for sd in seeds[:n_models]:
            params_run = dict(params)
            params_run["seed"] = int(sd)
            models.append(
                xgb.train(
                    params_run,
                    dtrain,
                    num_boost_round=3000,
                    evals=watchlist,
                    early_stopping_rounds=200 if dvalid is not None else None,
                    verbose_eval=100,
                )
            )

    result = {}
    valid_preds = []
    test_preds = []
    for m in models:
        if hasattr(m, "best_ntree_limit") and m.best_ntree_limit:
            best_ntree = m.best_ntree_limit
        elif hasattr(m, "best_iteration") and m.best_iteration:
            best_ntree = m.best_iteration
        else:
            best_ntree = m.num_boosted_rounds()

        if dvalid is not None:
            valid_preds.append(m.predict(dvalid, iteration_range=(0, best_ntree)))
        if test is not None:
            dtest = xgb.DMatrix(test[feature_cols])
            test_preds.append(m.predict(dtest, iteration_range=(0, best_ntree)))

    if valid_preds:
        result["valid_prob"] = np.mean(np.vstack(valid_preds), axis=0)
    if test_preds:
        result["test_result"] = pd.concat(
            [
                test[["User_id", "Coupon_id", "Date_received"]].copy(),
                pd.DataFrame(np.mean(np.vstack(test_preds), axis=0), columns=["prob"]),
            ],
            axis=1,
        )

    # 重要性（取最后一个模型）
    last_model = models[-1]
    score = last_model.get_score()
    feat_importance = pd.DataFrame(
        {"feature_name": list(score.keys()), "importance": list(score.values())}
    ).sort_values("importance", ascending=False)
    return result, feat_importance


def model_lgb(train, valid=None, test=None, n_models=1, seeds=None):
    """LightGBM模型，支持验证集与测试集预测"""
    if lgb is None:
        raise ImportError("LightGBM 未安装，请先 pip install lightgbm")

    feature_cols = [
        c
        for c in train.columns
        if c not in ["User_id", "Coupon_id", "Date_received", "label"]
    ]
    print(
        f"LightGBM 使用特征数: {len(feature_cols)}, 训练样本: {len(train)}, 验证样本: {len(valid) if valid is not None else 0}",
        flush=True,
    )
    dtrain = lgb.Dataset(train[feature_cols], label=train["label"])

    valid_sets = [dtrain]
    valid_names = ["train"]
    dvalid = None
    if valid is not None:
        dvalid = lgb.Dataset(valid[feature_cols], label=valid["label"])
        valid_sets.append(dvalid)
        valid_names.append("valid")

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "max_depth": 7,
        "min_data_in_leaf": 50,
        "lambda_l1": 1.0,
        "lambda_l2": 2.5,
        "min_gain_to_split": 0.01,
        "verbosity": -1,
    }

    callbacks = [lgb.log_evaluation(100)]
    if dvalid is not None:
        callbacks.append(lgb.early_stopping(200))

    # 多种子集成（不做扰动，避免线上不一致）
    if seeds is None or n_models <= 1:
        models = [
            lgb.train(
                params,
                dtrain,
                num_boost_round=3000,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
        ]
    else:
        models = []
        for sd in seeds[:n_models]:
            params_run = dict(params)
            params_run["seed"] = int(sd)
            models.append(
                lgb.train(
                    params_run,
                    dtrain,
                    num_boost_round=3000,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks,
                )
            )

    result = {}
    valid_preds = []
    test_preds = []
    for model in models:
        best_iter = (
            model.best_iteration
            if model.best_iteration is not None
            else model.current_iteration()
        )
        if dvalid is not None:
            valid_preds.append(
                model.predict(valid[feature_cols], num_iteration=best_iter)
            )
        if test is not None:
            test_preds.append(
                model.predict(test[feature_cols], num_iteration=best_iter)
            )
    if valid_preds:
        result["valid_prob"] = np.mean(np.vstack(valid_preds), axis=0)
    if test_preds:
        predict = pd.DataFrame(np.mean(np.vstack(test_preds), axis=0), columns=["prob"])
        result["test_result"] = pd.concat(
            [test[["User_id", "Coupon_id", "Date_received"]], predict], axis=1
        )

    last_model = models[-1]
    feat_importance = pd.DataFrame(
        {
            "feature_name": last_model.feature_name(),
            "importance": last_model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    return result, feat_importance


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "ccf_offline_stage1_train.csv"
    test_path = base_dir / "ccf_offline_stage1_test_revised.csv"
    online_path = base_dir / "ccf_online_stage1_train.csv"
    output_dir = base_dir / "output_files"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission_lgb.csv"

    # 源数据
    off_train = pd.read_csv(train_path, skipinitialspace=True)
    off_test = pd.read_csv(test_path, skipinitialspace=True)
    off_train.columns = off_train.columns.str.strip()
    off_test.columns = off_test.columns.str.strip()

    # 线上数据用于额外特征
    online_feat_dict = {}
    if online_path.exists():
        online_df = pd.read_csv(online_path, skipinitialspace=True)
        online_df.columns = online_df.columns.str.strip()
        try:
            online_feat_dict = build_online_features(online_df)
            print("已加载线上行为特征")
        except Exception as e:
            print(f"线上数据特征构建失败: {e}")

    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)

    # ==========================================
    # 滑窗构建数据集 (Sliding Window)
    # ==========================================

    # 辅助函数：构建切片
    def process_window(
        label_start, label_end, history_start, history_end, label_df=None
    ):
        """构建一个时间窗口的数据集"""
        if label_df is None:
            label_df = off_train

        # 标签区间
        label_field = label_df[
            label_df["date_received"].isin(pd.date_range(label_start, label_end))
        ]
        # 历史区间
        # 1. 领券记录 (date_received 在 history_start ~ history_end)
        cond_coupon = off_train["date_received"].isin(
            pd.date_range(history_start, history_end)
        )
        # 2. 普通消费记录 (date 在 history_start ~ history_end 且 date_received 为空)
        # 注意：prepare中 date_received 为 NaT
        cond_regular = (
            off_train["date"].isin(pd.date_range(history_start, history_end))
        ) & (off_train["date_received"].isnull())

        history_field = off_train[cond_coupon | cond_regular]
        # 中间区间（缓冲，防止穿越，这里设为空，通过日期控制）
        middle_field = pd.DataFrame()

        print(
            f"构建窗口: Label[{label_start}~{label_end}], History[{history_start}~{history_end}]"
        )
        print(f"Label样本数: {len(label_field)}, History样本数: {len(history_field)}")

        return get_dataset(
            history_field,
            middle_field,
            label_field,
            online_feats=online_feat_dict,
        )

    print("开始构建滑窗数据集...")

    # 窗口1: 20160416-20160515 (Label), 20160101-20160413 (History)
    # 对应 reference dataset1
    train_1 = process_window("2016/4/16", "2016/5/15", "2016/1/1", "2016/4/13")

    # 窗口2: 20160516-20160615 (Label), 20160201-20160514 (History)
    # 对应 reference dataset2
    train_2 = process_window("2016/5/16", "2016/6/15", "2016/2/1", "2016/5/14")

    # 测试集: 20160701-20160731 (Label), 20160315-20160630 (History)
    # 对应 reference dataset3
    test = process_window(
        "2016/7/1", "2016/7/31", "2016/3/15", "2016/6/30", label_df=off_test
    )

    # 合并训练集
    print("合并训练集...")
    train_full = pd.concat([train_1, train_2], axis=0)
    train_full.index = range(len(train_full))

    # 验证集策略：使用 train_2 的后半部分作为验证集，或者随机划分
    # 为了更接近线上分布，使用 train_2 (最近一个月) 作为验证集参考，但训练使用全量
    # 这里我们采用 train_1 + train_2 (前2周) 训练，train_2 (后2周) 验证
    # 或者简单点，随机划分，因为我们已经有了时间序列的构造
    # 更好的方式：Train on (Set1 + Set2), Validate on Set2 (to see recent performance)

    # 划分训练/验证
    # 方案：使用 train_2 作为验证集来评估模型（因为它最接近测试集），但最终预测使用全量训练
    validate = train_2.copy()
    train = train_full.copy()  # 全量训练

    print(
        f"最终训练集样本数: {len(train)}, 验证集样本数: {len(validate)}, 测试集样本数: {len(test)}"
    )

    # 线下验证（若可用）
    lgb_offline_auc = None
    xgb_offline_auc = None
    lgb_valid_result = None
    xgb_valid_result = None

    if lgb is not None:
        try:
            print("开始 LightGBM 线下验证 (使用最近一个月数据作为验证)...", flush=True)
            # 注意：这里为了验证效果，我们应该只用 train_1 训练，train_2 验证
            # 但为了最终提交，我们希望用更多数据。
            # 折中：先用 train_1 训, train_2 验，看分数；然后用 train_full 训，预测 test

            # 验证阶段
            lgb_valid_model, _ = model_lgb(
                train_1, validate, None, n_models=1, seeds=[2024]
            )
            validate_pred = validate[
                ["User_id", "Coupon_id", "Date_received", "label"]
            ].copy()
            validate_pred["prob"] = lgb_valid_model["valid_prob"]
            lgb_offline_auc = evaluate_coupon_auc(validate_pred)
            print(
                f"LightGBM 线下按券平均AUC (Train:Set1, Valid:Set2): {lgb_offline_auc:.4f}"
            )
        except Exception as e:
            print(f"LightGBM 线下验证失败: {e}")

    try:
        print("开始 XGBoost 线下验证 (使用最近一个月数据作为验证)...", flush=True)
        xgb_valid_model, _ = model_xgb(
            train_1, validate, None, n_models=1, seeds=[2024]
        )
        validate_pred_xgb = validate[
            ["User_id", "Coupon_id", "Date_received", "label"]
        ].copy()
        validate_pred_xgb["prob"] = xgb_valid_model["valid_prob"]
        xgb_offline_auc = evaluate_coupon_auc(validate_pred_xgb)
        print(
            f"XGBoost 线下按券平均AUC (Train:Set1, Valid:Set2): {xgb_offline_auc:.4f}"
        )
    except Exception as e:
        print(f"XGBoost 线下验证失败: {e}")

    # ==========================================
    # 5-Fold Stacking
    # ==========================================
    from sklearn.model_selection import StratifiedKFold

    print("开始 5-Fold Stacking...", flush=True)

    # 准备数据
    feature_cols = [
        c
        for c in train_full.columns
        if c not in ["User_id", "Coupon_id", "Date_received", "label"]
    ]
    X = train_full[feature_cols]
    y = train_full["label"]
    X_test = test[feature_cols]

    # 存储 OOF 预测和测试集预测
    oof_lgb = np.zeros(len(train_full))
    oof_xgb = np.zeros(len(train_full))
    test_pred_lgb = np.zeros(len(test))
    test_pred_xgb = np.zeros(len(test))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2024)

    # LightGBM CV
    if lgb is not None:
        print("Running LightGBM CV...", flush=True)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val)

            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.01,
                "num_leaves": 31,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "max_depth": 7,
                "min_data_in_leaf": 50,
                "lambda_l1": 1.0,
                "lambda_l2": 2.5,
                "min_gain_to_split": 0.01,
                "verbosity": -1,
                "seed": 2024 + fold,
            }

            model = lgb.train(
                params,
                dtrain,
                num_boost_round=3000,
                valid_sets=[dtrain, dval],
                callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)],
            )

            oof_lgb[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            test_pred_lgb += (
                model.predict(X_test, num_iteration=model.best_iteration) / 5
            )
            print(
                f"Fold {fold+1} LGB AUC: {roc_auc_score(y_val, oof_lgb[val_idx]):.4f}"
            )

    # XGBoost CV
    print("Running XGBoost CV...", flush=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval = xgb.DMatrix(X_val, label=y_val)
        dtest = xgb.DMatrix(X_test)

        params = {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.01,
            "max_depth": 5,
            "min_child_weight": 1.1,
            "gamma": 0.1,
            "lambda": 10,
            "colsample_bylevel": 0.7,
            "colsample_bytree": 0.7,
            "subsample": 0.7,
            "scale_pos_weight": 1,
            "verbosity": 0,
            "seed": 2024 + fold,
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=200,
            verbose_eval=False,
        )

        oof_xgb[val_idx] = model.predict(
            dval, iteration_range=(0, model.best_iteration + 1)
        )
        test_pred_xgb += (
            model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / 5
        )
        print(f"Fold {fold+1} XGB AUC: {roc_auc_score(y_val, oof_xgb[val_idx]):.4f}")

    # Stacking (Logistic Regression)
    print("Training Meta-Learner (Logistic Regression)...", flush=True)
    X_stack_train = np.vstack([oof_lgb, oof_xgb]).T
    X_stack_test = np.vstack([test_pred_lgb, test_pred_xgb]).T

    meta_model = LogisticRegression()
    meta_model.fit(X_stack_train, y)

    stack_pred = meta_model.predict_proba(X_stack_test)[:, 1]

    # 生成提交
    submission = test[["User_id", "Coupon_id", "Date_received"]].copy()
    submission["prob"] = stack_pred
    submission = dedup_result(submission)

    submission.to_csv(output_path, index=False, header=None)
    print(f"保存 Stacking 提交文件: {output_path}")
    print(
        f"Meta-Learner Coefficients: LGB={meta_model.coef_[0][0]:.4f}, XGB={meta_model.coef_[0][1]:.4f}"
    )

    # 备份线性融合结果
    blended_prob = 0.5 * test_pred_lgb + 0.5 * test_pred_xgb
    submission_blend = test[["User_id", "Coupon_id", "Date_received"]].copy()
    submission_blend["prob"] = blended_prob
    submission_blend = dedup_result(submission_blend)
    submission_blend.to_csv(
        str(output_path).replace(".csv", "_blend.csv"), index=False, header=None
    )
    print("保存线性融合备份: submission_lgb_blend.csv")
