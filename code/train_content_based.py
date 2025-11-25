# train_content_based.py
# -*- coding: utf-8 -*-
import argparse
import os
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression  # 작은 회귀
from tqdm.auto import tqdm

# from core_content_based import (
#     ContentBasedModel, FACTOR_COLS, TARGET_COL_R, TARGET_COL_C
# )

from test_core import (
    ContentBasedModel, FACTOR_COLS, TARGET_COL_R, TARGET_COL_C
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="path to preprocessed_2.csv")
    parser.add_argument("--out",   required=True, help="output .pkl path")
    parser.add_argument("--alpha", type=float, default=20.0, help="Bayesian smoothing strength")
    parser.add_argument("--weight-mode", choices=["sim","rating"], default="rating",
                        help="weighted mean per item: uniform or rating-weighted")
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    required = {"user_id","item_id", *FACTOR_COLS, TARGET_COL_R, TARGET_COL_C}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in train: {miss}")

    model = ContentBasedModel(alpha=args.alpha, weight_mode=args.weight_mode)

    # 1) 공통 좌표계(스케일러) 적합: 원시 factor 전체
    model.fit_scaler(df)

    # 2) Create item profile with bayesian smoothing
    model.build_item_profiles(df)

    # 3) 아이템 평균 평점(bias) 구축  ← 회귀 입력 feature에 사용
    model.build_item_bias(df)

    # 3) Make calibration data with (seed,target) pair
    X, y = model.make_pair_features(df)
    if len(X) == 0:
        raise RuntimeError("No training pairs for calibration. Check data and item_profiles.")
    
    # fit classifier
    model.fit_rec_threshold(df, mode="sim")      # 또는 mode="rating"


    # 5) 작은 회귀 학습: r_hat = a*sim + b*seed + c*item_bias + d
    reg = LinearRegression()
    reg.fit(X, y)
    model.calib_coef = reg.coef_.astype(float)          # [a,b,c]
    model.calib_intercept = float(reg.intercept_)       # d

    # 6) 저장
    out_path = f"{args.out}_alpha{int(args.alpha)}_{args.weight_mode}.pkl"
    # os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        cloudpickle.dump(model, f)
    # out_path = f"{out}_{k}_{shrink}_{ridge}.pkl"
    # with open(out_path, "wb") as f:
    #     cloudpickle.dump(model, f)

    print(f"[OK] Saved content-based model to {args.out}")
    print(f"   - alpha={args.alpha}, weight_mode={args.weight_mode}")
    print(f"   - #item_profiles: {len(model.item_profiles)}")
    print(f"   - calib coef [a, b, c]: {model.calib_coef}, intercept d: {model.calib_intercept:.4f}")

if __name__ == "__main__":
    main()
