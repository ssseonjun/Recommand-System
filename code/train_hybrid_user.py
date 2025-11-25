# train_hybrid_user_cf_no_profile.py
# -*- coding: utf-8 -*-
import argparse
import itertools
import os
import cloudpickle  # pip install cloudpickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from core_hybrid_user import (
    HybridUserCFNoProfile, RANDOM_SEED, FACTOR_COLS, TARGET_COL_R, TARGET_COL_C
)

def train_and_save(df: pd.DataFrame, out_prefix: str,
                   k: int, shrink: float, ridge: float, folds: int):
    model = HybridUserCFNoProfile(
        neighbour_k=k, shrink_alpha=shrink, ridge_alpha=ridge, content_weight=0.0
    )
    # 1) 스케일러: train의 raw factor 전체로 적합
    model.fit_scaler_on_factors(df)
    # 2) 사용자 선호(가중치) 학습
    model.fit_user_preferences_train(df)
    # 3) K-fold로 Recommended 분류기 학습(Δ/ρ)
    model.train_classifier_with_kfold(df, n_splits=folds)

    out_path = f"{out_prefix}_{k}_{shrink}_{ridge}.pkl"
    with open(out_path, "wb") as f:
        cloudpickle.dump(model, f)
    print(f"[OK] Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="path to preprocessed_2.csv")
    # out은 접두어(prefix). 여러 모델을 저장하므로 자동으로 접미사가 붙는다.
    parser.add_argument("--out", required=True, help="output file prefix (no extension)")
    # 여러 개 값을 받을 수 있게 nargs='+'
    parser.add_argument("--k", nargs="+", type=int, default=[100], help="top-K neighbours (one or more)")
    parser.add_argument("--shrink", nargs="+", type=float, default=[20.0], help="similarity shrinkage alpha (one or more)")
    parser.add_argument("--ridge", nargs="+", type=float, default=[5.0], help="ridge alpha for user prefs (one or more)")
    parser.add_argument("--folds", type=int, default=3, help="K-fold for rec classifier")
    args = parser.parse_args()

    df = pd.read_csv(args.train)
    required = {"user_id","item_id", *FACTOR_COLS, TARGET_COL_R, TARGET_COL_C}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in train: {miss}")

    # 그리드 조합 생성
    combos = list(itertools.product(args.k, args.shrink, args.ridge))
    print(f"[INFO] Will train {len(combos)} model(s):")
    for K, SH, RG in combos:
        print(f"  - k={K}, shrink={SH}, ridge={RG}")

    # 학습 루프
    for K, SH, RG in tqdm(combos, desc="Training sweep"):
        train_and_save(df, args.out, K, SH, RG, args.folds)

if __name__ == "__main__":
    main()
