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
    # 距离处理
    data["Distance"].fillna(-1, inplace=True)  # 空距离填充为-1
    data["null_distance"] = data["Distance"].map(lambda x: 1 if x == -1 else 0)
    # 时间处理
    data["date_received"] = pd.to_datetime(data["Date_received"], format="%Y%m%d")
    if "Date" in data.columns.tolist():  # off_train
        data["date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
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


def build_history_features(history_field):
    """从历史区间提取用户/商家/优惠券/交互统计特征"""
    if history_field.empty:
        return {}, pd.DataFrame()

    data = history_field.copy()
    data["used"] = data["Date"].notnull().astype(int)

    # 用户维度
    user_grp = data.groupby("User_id")
    user_feat = pd.DataFrame({"User_id": user_grp.size().index})
    user_feat["user_receive_cnt"] = user_grp.size().values
    user_feat["user_used_cnt"] = user_grp["used"].sum().values
    user_feat["user_used_rate"] = safe_divide(
        user_feat["user_used_cnt"], user_feat["user_receive_cnt"]
    )
    user_feat["user_merchant_cnt"] = user_grp["Merchant_id"].nunique().values
    user_feat["user_coupon_cnt"] = user_grp["Coupon_id"].nunique().values
    user_feat["user_distance_mean"] = (
        user_grp["Distance"].apply(lambda x: x.replace(-1, np.nan).mean()).values
    )
    user_feat["user_distance_min"] = (
        user_grp["Distance"].apply(lambda x: x.replace(-1, np.nan).min()).values
    )
    user_feat["user_distance_max"] = (
        user_grp["Distance"].apply(lambda x: x.replace(-1, np.nan).max()).values
    )
    user_feat.fillna(0, inplace=True)

    # 商家维度
    merchant_grp = data.groupby("Merchant_id")
    merchant_feat = pd.DataFrame({"Merchant_id": merchant_grp.size().index})
    merchant_feat["merchant_receive_cnt"] = merchant_grp.size().values
    merchant_feat["merchant_used_cnt"] = merchant_grp["used"].sum().values
    merchant_feat["merchant_used_rate"] = safe_divide(
        merchant_feat["merchant_used_cnt"], merchant_feat["merchant_receive_cnt"]
    )
    merchant_feat["merchant_user_cnt"] = merchant_grp["User_id"].nunique().values
    merchant_feat["merchant_coupon_cnt"] = merchant_grp["Coupon_id"].nunique().values

    # 优惠券维度
    coupon_grp = data.groupby("Coupon_id")
    coupon_feat = pd.DataFrame({"Coupon_id": coupon_grp.size().index})
    coupon_feat["coupon_receive_cnt"] = coupon_grp.size().values
    coupon_feat["coupon_used_cnt"] = coupon_grp["used"].sum().values
    coupon_feat["coupon_used_rate"] = safe_divide(
        coupon_feat["coupon_used_cnt"], coupon_feat["coupon_receive_cnt"]
    )

    # 用户-商家交互
    um_grp = data.groupby(["User_id", "Merchant_id"])
    um_feat = pd.DataFrame(
        {
            "User_id": [u for u, _ in um_grp.size().index],
            "Merchant_id": [m for _, m in um_grp.size().index],
        }
    )
    um_feat["um_receive_cnt"] = um_grp.size().values
    um_feat["um_used_cnt"] = um_grp["used"].sum().values
    um_feat["um_used_rate"] = safe_divide(
        um_feat["um_used_cnt"], um_feat["um_receive_cnt"]
    )

    return {
        "user": user_feat,
        "merchant": merchant_feat,
        "coupon": coupon_feat,
        "um": um_feat,
    }, data[["User_id", "Coupon_id", "Date_received", "Date"]]


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
    return tmp[["row_id", "user_gap_since_last", "user_gap_until_next"]]


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


def get_dataset(history_field, middle_field, label_field):
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
    history_feats, _ = build_history_features(history_field)

    # 构造数据集
    share_characters = list(
        set(simple_feat.columns.tolist()) & set(week_feat.columns.tolist())
    )
    dataset = pd.concat([week_feat, simple_feat.drop(share_characters, axis=1)], axis=1)
    dataset = dataset.merge(recency_feat, on="row_id", how="left")

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

    # 删除无用属性并将label置于最后一列
    if "Date" in dataset.columns.tolist():  # 表示训练集和验证集
        dataset.drop(
            ["Merchant_id", "Discount_rate", "Date", "date_received", "date"],
            axis=1,
            inplace=True,
        )
        label = dataset["label"].tolist()
        dataset.drop(["label"], axis=1, inplace=True)
        dataset["label"] = label
    else:  # 表示测试集
        dataset.drop(
            ["Merchant_id", "Discount_rate", "date_received"], axis=1, inplace=True
        )

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


def model_xgb(train, test):
    """xgb模型

    Args:

    Returns:

    """
    # xgb参数
    params = {
        "booster": "gbtree",
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "silent": 1,
        "eta": 0.01,
        "max_depth": 5,
        "min_child_weight": 1,
        "gamma": 0,
        "lambda": 1,
        "colsample_bylevel": 0.7,
        "colsample_bytree": 0.7,
        "subsample": 0.9,
        "scale_pos_weight": 1,
    }
    # 数据集
    dtrain = xgb.DMatrix(
        train.drop(["User_id", "Coupon_id", "Date_received", "label"], axis=1),
        label=train["label"],
    )
    dtest = xgb.DMatrix(test.drop(["User_id", "Coupon_id", "Date_received"], axis=1))
    # 训练
    watchlist = [(dtrain, "train")]
    model = xgb.train(params, dtrain, num_boost_round=5167, evals=watchlist)
    # 预测
    predict = model.predict(dtest)
    # 处理结果
    predict = pd.DataFrame(predict, columns=["prob"])
    result = pd.concat(
        [test[["User_id", "Coupon_id", "Date_received"]], predict], axis=1
    )
    # 特征重要性
    feat_importance = pd.DataFrame(columns=["feature_name", "importance"])
    feat_importance["feature_name"] = model.get_score().keys()
    feat_importance["importance"] = model.get_score().values()
    feat_importance.sort_values(["importance"], ascending=False, inplace=True)
    # 返回
    return result, feat_importance


def model_lgb(train, valid=None, test=None):
    """LightGBM模型，支持验证集与测试集预测"""
    if lgb is None:
        raise ImportError("LightGBM 未安装，请先 pip install lightgbm")

    feature_cols = [
        c
        for c in train.columns
        if c not in ["User_id", "Coupon_id", "Date_received", "label"]
    ]
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
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "min_data_in_leaf": 50,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1500,
        valid_sets=valid_sets,
        valid_names=valid_names,
        early_stopping_rounds=100 if dvalid is not None else None,
    )

    result = {}
    if dvalid is not None:
        result["valid_prob"] = model.predict(
            valid[feature_cols], num_iteration=model.best_iteration
        )
    if test is not None:
        test_prob = model.predict(
            test[feature_cols], num_iteration=model.best_iteration
        )
        predict = pd.DataFrame(test_prob, columns=["prob"])
        result["test_result"] = pd.concat(
            [test[["User_id", "Coupon_id", "Date_received"]], predict], axis=1
        )

    feat_importance = pd.DataFrame(
        {
            "feature_name": model.feature_name(),
            "importance": model.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    return result, feat_importance


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "ccf_offline_stage1_train.csv"
    test_path = base_dir / "ccf_offline_stage1_test_revised.csv"
    output_dir = base_dir / "output_files"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission_lgb.csv"

    # 源数据
    off_train = pd.read_csv(train_path)
    off_test = pd.read_csv(test_path)

    # 预处理
    off_train = prepare(off_train)
    off_test = prepare(off_test)
    # 打标
    off_train = get_label(off_train)

    # 划分区间
    train_history_field = off_train[
        off_train["date_received"].isin(pd.date_range("2016/3/2", periods=60))
    ]
    train_middle_field = off_train[
        off_train["date"].isin(pd.date_range("2016/5/1", periods=15))
    ]
    train_label_field = off_train[
        off_train["date_received"].isin(pd.date_range("2016/5/16", periods=31))
    ]

    validate_history_field = off_train[
        off_train["date_received"].isin(pd.date_range("2016/1/16", periods=60))
    ]
    validate_middle_field = off_train[
        off_train["date"].isin(pd.date_range("2016/3/16", periods=15))
    ]
    validate_label_field = off_train[
        off_train["date_received"].isin(pd.date_range("2016/3/31", periods=31))
    ]

    test_history_field = off_train[
        off_train["date_received"].isin(pd.date_range("2016/4/17", periods=60))
    ]
    test_middle_field = off_train[
        off_train["date"].isin(pd.date_range("2016/6/16", periods=15))
    ]
    test_label_field = off_test.copy()

    # 构造训练集、验证集、测试集
    print("构造训练集")
    train = get_dataset(train_history_field, train_middle_field, train_label_field)
    print("构造验证集")
    validate = get_dataset(
        validate_history_field, validate_middle_field, validate_label_field
    )
    print("构造测试集")
    test = get_dataset(test_history_field, test_middle_field, test_label_field)

    # 线下验证（若可用）
    if lgb is not None:
        try:
            lgb_valid_result, _ = model_lgb(train, validate, None)
            validate_pred = validate[
                ["User_id", "Coupon_id", "Date_received", "label"]
            ].copy()
            validate_pred["prob"] = lgb_valid_result["valid_prob"]
            offline_auc = evaluate_coupon_auc(validate_pred)
            print(f"线下按券平均AUC: {offline_auc:.4f}")
        except Exception as e:
            print(f"LightGBM 线下验证失败: {e}")
            lgb_valid_result = None
    else:
        lgb_valid_result = None

    # 线上训练与预测
    big_train = pd.concat([train, validate], axis=0)
    try:
        if lgb is None:
            raise ImportError("LightGBM 未安装")
        final_result, feat_importance = model_lgb(big_train, None, test)
        submission = dedup_result(final_result["test_result"])
        submission.to_csv(output_path, index=False, header=None)
        print(f"保存 LightGBM 提交文件: {output_path}")
    except Exception as e:
        print(f"LightGBM 训练失败，回退 XGBoost: {e}")
        result, feat_importance = model_xgb(big_train, test)
        submission = dedup_result(result)
        submission.to_csv(output_path, index=False, header=None)
        print(f"保存 XGBoost 提交文件: {output_path}")
