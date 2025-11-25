# evaluate_models.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, pickle, random
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os, pickle

RANDOM_SEED = 42
FACTOR_COLS = ["Class","Seat Comfort","Staff Service","Food & Beverages","Inflight Entertainment","Value For Money"]
TARGET_COL_R = "Overall Rating"
TARGET_COL_C = "Recommended"

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)

try:
    import cloudpickle
    _HAS_CLOUD = True
except Exception:
    _HAS_CLOUD = False

def load_model(path: str):
    """cloudpickle 우선, 실패 시 pickle로 모델(.pkl) 로드"""
    with open(path, "rb") as f:
        if _HAS_CLOUD:
            try:
                return cloudpickle.load(f)
            except Exception:
                pass
        f.seek(0)
        return pickle.load(f)

def _build_test_item_profile_global(df_test: pd.DataFrame):
    sums = defaultdict(lambda: np.zeros(len(FACTOR_COLS), dtype=float))
    cnts = Counter()
    ui_factor = {}
    for _, r in df_test.iterrows():
        uid, iid = int(r["user_id"]), int(r["item_id"])
        v = r[FACTOR_COLS].to_numpy(dtype=float)
        sums[iid] += v; cnts[iid] += 1
        ui_factor[(uid,iid)] = v
    return sums, cnts, ui_factor

def _louo(iid:int, uid:int, sums:Dict[int,np.ndarray], cnts:Dict[int,int], ui_factor:Dict[Tuple[int,int],np.ndarray], scaler) -> Optional[np.ndarray]:
    if iid not in cnts or cnts[iid] <= 0: return None
    s = sums[iid].copy(); n = cnts[iid]
    key = (uid,iid)
    if key in ui_factor:
        s -= ui_factor[key]; n -= 1
    if n <= 0: return None
    v = (s / n).reshape(1,-1)
    return scaler.transform(v).ravel() if scaler is not None else v.ravel()

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v/(n if n>0 else 1.0)

def _shrink(n: int, alpha: float) -> float:
    return n / (n + alpha)

def _sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

def evaluate_one_model(model_path: str, df_test: pd.DataFrame, neighbour_k: int = None) -> Dict[str, float]:
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # allow override neighbour_k
    KNEI = neighbour_k if neighbour_k is not None else model.neighbour_k
    γ = float(getattr(model, "content_weight", 0.0))

    # Build global profiles for test (for neighbour baselines)
    sums, cnts, ui_factor = _build_test_item_profile_global(df_test)
    lookup_test_global = {}
    for iid in cnts:
        if cnts[iid] > 0:
            vec = (sums[iid] / cnts[iid]).reshape(1,-1)
            lookup_test_global[iid] = model.scaler.transform(vec).ravel() if model.scaler is not None else vec.ravel()

    # Fit test users' w_v, b_v on full test (for neighbour pool)
    user_w_test, user_b_test, user_n_test = {}, {}, {}
    for uid, g in tqdm(df_test.groupby("user_id"), desc="Test: fit user w", total=df_test["user_id"].nunique()):
        X, y = [], []
        for _, row in g.iterrows():
            x = lookup_test_global.get(int(row["item_id"]))
            if x is None: continue
            X.append(x); y.append(float(row[TARGET_COL_R]))
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
        if len(y) >= 2:
            from sklearn.linear_model import Ridge
            m = Ridge(alpha=model.ridge_alpha, fit_intercept=True, random_state=RANDOM_SEED)
            m.fit(X, y)
            w = m.coef_.astype(float); b = float(m.intercept_)
        elif len(y) == 1:
            w = np.zeros(len(FACTOR_COLS), dtype=float); b = float(y[0])
        else:
            w = np.zeros(len(FACTOR_COLS), dtype=float); b = 0.0
        user_w_test[uid], user_b_test[uid], user_n_test[uid] = w, b, len(y)

    # pre-normalized for cosine
    W_train_norm = {u: _norm(w) for u, w in model.user_w_train.items()}
    W_test_norm  = {u: _norm(w) for u, w in user_w_test.items()}

    # labels & index
    rec_label = {(int(r.user_id), int(r.item_id)): 1 if r[TARGET_COL_C] in (True,1,"True","true") else 0
                 for _, r in df_test.iterrows()}
    rating_label = {(int(r.user_id), int(r.item_id)): float(r[TARGET_COL_R]) for _, r in df_test.iterrows()}
    user_items = defaultdict(list)
    for _, r in df_test.iterrows():
        user_items[int(r.user_id)].append(int(r.item_id))

    # accumulators (micro)
    all_y_true_cls, all_y_pred_cls = [], []
    all_true_r, all_pred_r = [], []

    # per-user loop (only users with >=3 items)
    users = sorted(df_test["user_id"].unique().tolist())
    for u in tqdm(users, desc="Evaluate (seed=all, min 3)"):
        items_u = user_items[u]
        if len(items_u) < 3: 
            continue

        # choose seeds = all items
        for seed in items_u:
            y_seed = rating_label[(u, seed)]
            x_seed = _louo(seed, u, sums, cnts, ui_factor, model.scaler)

            # prior from train-global
            w_prior = model.global_w.copy()
            b_prior = model.global_b
            # weak 1-shot update (closed-form)
            if x_seed is not None:
                denom = model.ridge_alpha + float(x_seed.dot(x_seed))
                adj = (y_seed - b_prior - float(w_prior.dot(x_seed))) / denom
                w_u = w_prior + adj * x_seed
                b_u = (b_prior*model.ridge_alpha + y_seed) / (model.ridge_alpha + 1.0)
            else:
                w_u, b_u = w_prior, b_prior

            # neighbours (train + test_except_u)
            neigh = []
            for v, wv in W_train_norm.items():
                s = float(w_u.dot(wv))
                if s <= 0: continue
                s *= _shrink(model.user_n_train.get(v,0), model.shrink_alpha) * _shrink(1, model.shrink_alpha)
                if s > 0: neigh.append((("train",v), s))
            for v, wv in W_test_norm.items():
                if v == u: continue
                s = float(w_u.dot(wv))
                if s <= 0: continue
                s *= _shrink(user_n_test.get(v,0), model.shrink_alpha) * _shrink(1, model.shrink_alpha)
                if s > 0: neigh.append((("test",v), s))
            neigh.sort(key=lambda x: -x[1])
            neigh = neigh[:KNEI]

            # predict for the rest (exclude THIS seed)
            candidates = [i for i in items_u if i != seed]
            for i in candidates:
                x_i = _louo(i, u, sums, cnts, ui_factor, model.scaler)
                base = float(w_u.dot(x_i)) if x_i is not None else 0.0
                r0 = b_u + γ*base

                # CF features from neighbours who rated i (TEST domain ratings only)
                num, den = 0.0, 0.0
                num_res, den_res = 0.0, 0.0
                for (domain,v), s in neigh:
                    if domain == "test" and (v,i) in rating_label:
                        den += abs(s)
                        num += s * (1 if rec_label[(v,i)]==1 else 0)
                        x_vi = lookup_test_global.get(i)
                        r0_v = user_b_test.get(v,0.0) + γ*(float(user_w_test.get(v, np.zeros_like(w_u)).dot(x_vi)) if x_vi is not None else 0.0)
                        num_res += s * (rating_label[(v,i)] - r0_v)
                        den_res += abs(s)
                rho = (num/den) if den>0 else 0.0
                delta = (num_res/den_res) if den_res>0 else 0.0

                # rec probability
                if model.clf is not None:
                    Xf = np.array([[r0, delta, rho, b_u, 0.0]], dtype=float)
                    p_rec = float(model.clf.predict_proba(Xf)[0,1])
                else:
                    p_rec = _sigmoid(r0 + delta + 2.0*rho)

                y_true = rec_label[(u,i)]
                y_pred = 1 if p_rec >= 0.5 else 0

                all_y_true_cls.append(y_true)
                all_y_pred_cls.append(y_pred)

                # rating (aux RMSE)
                r_hat = r0 + delta
                all_true_r.append(float(rating_label[(u,i)]))
                all_pred_r.append(float(r_hat))

    # compute metrics (micro)
    tp = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==1 and yp==1)
    fp = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==1 and yp==0)
    tn = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==0 and yp==0)
    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    acc  = (tp+tn) / max(tp+tn+fp+fn, 1)
    f1   = (2*prec*rec) / max(prec+rec, 1e-12)

    # rmse = math.sqrt(mean_absolute_error(all_true_r, all_pred_r)) if all_true_r else float("nan")
    mae = mean_absolute_error(all_true_r, all_pred_r) if all_true_r else float("nan")

    return {
        "Precision": prec,
        "Recall": rec,
        "Accuracy": acc,
        "F1": f1,
        "Samples": len(all_y_true_cls)
    }

# --- Content-based helpers ---
def _cb_build_test_profiles_LOUO(test_df: pd.DataFrame,
                                 scaler,
                                 mu_vec: np.ndarray,
                                 alpha: float,
                                 weight_mode: str):
    """
    테스트셋에서 아이템별 LOUO 프로파일을 만들기 위한 전처리 페이로드:
    - weight_mode="rating" 이면 평점 기반 가중 평균,
      그 외에는 단순 평균.
    - sums[iid] : Σ w_ui * v_ui (raw factor)
    - cnts[iid] : Σ w_ui (weight 합)
    - ui_raw[(u,i)] : (v_ui, w_ui)
    """
    from collections import defaultdict, Counter
    sums = defaultdict(lambda: np.zeros(len(FACTOR_COLS), dtype=float))
    cnts = Counter()
    ui_raw = {}

    for _, r in test_df.iterrows():
        uid, iid = int(r["user_id"]), int(r["item_id"])
        v_raw = r[FACTOR_COLS].to_numpy(dtype=float)

        # --- NEW: weight_mode 반영 ---
        if weight_mode == "rating":
            w = float(r[TARGET_COL_R])
            if not np.isfinite(w) or w < 0.0:
                w = 0.0
        else:
            w = 1.0

        sums[iid] += w * v_raw
        cnts[iid] += w
        ui_raw[(uid, iid)] = (v_raw, w)

    return sums, cnts, ui_raw


def _cb_louo_profile_for(test_uid: int,
                         iid: int,
                         sums,
                         cnts,
                         ui_raw,
                         scaler,
                         mu_vec: np.ndarray,
                         alpha: float,
                         weight_mode: str):
    """
    (u,i) 예측 시, u의 기여를 제외한 아이템 i의 평균 벡터를 만든 뒤
    scaled space로 보정하고, Empirical Bayes 스무딩까지 적용.

    - weight_mode="rating"이면, 각 관측치에 대한 weight(평점)를 사용해서
      Σ w*v / Σ w 형태의 가중 평균을 사용.
    """
    if iid not in cnts or cnts[iid] <= 0:
        return None

    s = sums[iid].copy()
    n = cnts[iid]
    key = (test_uid, iid)

    if key in ui_raw:
        v_raw, w = ui_raw[key]
        if weight_mode == "rating":
            s -= w * v_raw
            n -= w
        else:
            s -= v_raw
            n -= 1

    if n <= 0:
        return None

    mean_raw = (s / n).reshape(1, -1)
    mean_scaled = scaler.transform(mean_raw).ravel() if scaler is not None else mean_raw.ravel()

    a = float(alpha)
    prof = (n / (n + a)) * mean_scaled + (a / (n + a)) * mu_vec
    return prof.astype(float)


def evaluate_one_model_content_based(model, df_test: pd.DataFrame, seed_mode: str = "all") -> Dict[str, float]:
    """
    Content-based 평가 (evaluate_one_model의 구조/반환 스키마에 맞춤):
    - per-user 루프: 최소 3개 아이템 보유 사용자만 평가
    - seed=all(기본): 각 시드를 기준으로 나머지 아이템 예측
    - LOUO 아이템 프로파일(스케일+베이지안 스무딩) 사용
    - 평점: r_hat = calibrate(sim, seed_rating, item_bias)
    - 추천: seed Recommended에 따라 서로 다른 τ 패턴 사용
    """
    # 필수 속성
    scaler = getattr(model, "scaler", None)
    mu_vec = getattr(model, "mu_vec", None)
    alpha  = float(getattr(model, "alpha", 20.0))
    weight_mode = getattr(model, "weight_mode", "rating")

    # LOUO용 프리컴퓨트 (weight_mode 반영)
    sums, cnts, ui_raw = _cb_build_test_profiles_LOUO(df_test, scaler, mu_vec, alpha, weight_mode)

    # 라벨/인덱스
    rec_label = {(int(r.user_id), int(r.item_id)): 1 if r[TARGET_COL_C] in (True,1,"True","true") else 0
                 for _, r in df_test.iterrows()}
    rating_label = {(int(r.user_id), int(r.item_id)): float(r[TARGET_COL_R])
                    for _, r in df_test.iterrows()}
    user_items = defaultdict(list)
    for _, r in df_test.iterrows():
        user_items[int(r.user_id)].append(int(r.item_id))

    # 누적 (micro)
    all_y_true_cls, all_y_pred_cls = [], []
    all_true_r, all_pred_r = [], []

    # per-user loop (seed=all, min 3)
    users = sorted(df_test["user_id"].unique().tolist())
    for u in tqdm(users, desc="Evaluate (seed=all, min 3)"):
        items_u = user_items[u]
        if len(items_u) < 3:
            continue

        # seeds
        if seed_mode == "all":
            seeds = items_u
        else:
            rng = random.Random(RANDOM_SEED + u)
            seeds = [rng.choice(items_u)]

        for seed in seeds:
            # --- NEW: seed의 Recommended 확인 (패턴 구분용) ---
            seed_rec = rec_label[(u, seed)]  # 1 or 0

            # 시드 기준 사용자 벡터 (LOUO, scaled + EB smoothing)
            user_vec = _cb_louo_profile_for(u, seed, sums, cnts, ui_raw, scaler, mu_vec, alpha, weight_mode)
            if user_vec is None:
                continue
            y_seed = rating_label[(u, seed)]

            # 후보 = 시드 제외
            candidates = [i for i in items_u if i != seed]
            for i in candidates:
                item_vec = _cb_louo_profile_for(u, i, sums, cnts, ui_raw, scaler, mu_vec, alpha, weight_mode)
                if item_vec is None:
                    continue

                # 코사인 유사도
                nu = np.linalg.norm(user_vec); nv = np.linalg.norm(item_vec)
                sim = float(np.dot(user_vec, item_vec) / (nu*nv)) if nu>0 and nv>0 else 0.0
                sim = max(min(sim, 1.0), -1.0)

                # 평점 예측 (calibrate 사용)
                r_hat = model.predict_calibrated(sim, y_seed, i)
                all_true_r.append(float(rating_label[(u, i)]))
                all_pred_r.append(float(r_hat))

                # --- NEW: seed_rec에 따라 τ 패턴 선택 ---
                if getattr(model, "rec_decision", "sim") == "sim":
                    # 기본 전역 τ
                    thr = getattr(model, "tau_sim", 0.5)
                    # seed=1 / 0 별 패턴 τ가 있으면 사용
                    tau_pos = getattr(model, "tau_sim_pos", None)
                    tau_neg = getattr(model, "tau_sim_neg", None)
                    if seed_rec == 1 and tau_pos is not None:
                        thr = tau_pos
                    elif seed_rec == 0 and tau_neg is not None:
                        thr = tau_neg
                    y_pred = 1 if sim >= thr else 0
                else:  # rating 임계 사용
                    thr = getattr(model, "tau_rating", 5.0)
                    tau_pos = getattr(model, "tau_rating_pos", None)
                    tau_neg = getattr(model, "tau_rating_neg", None)
                    if seed_rec == 1 and tau_pos is not None:
                        thr = tau_pos
                    elif seed_rec == 0 and tau_neg is not None:
                        thr = tau_neg
                    y_pred = 1 if r_hat >= thr else 0

                all_y_true_cls.append(rec_label[(u, i)])
                all_y_pred_cls.append(y_pred)

    # 지표 계산
    tp = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==1 and yp==1)
    fp = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==1 and yp==0)
    tn = sum(1 for yt, yp in zip(all_y_true_cls, all_y_pred_cls) if yt==0 and yp==0)

    prec = tp / max(tp+fp, 1)
    rec  = tp / max(tp+fn, 1)
    acc  = (tp+tn) / max(tp+tn+fp+fn, 1)
    f1   = (2*prec*rec) / max(prec+rec, 1e-12)

    rmse = math.sqrt(mean_squared_error(all_true_r, all_pred_r)) if all_true_r else float("nan")
    mae = mean_absolute_error(all_true_r, all_pred_r) if all_true_r else float("nan")

    return {
        "MAE": mae,
        "Precision": prec,
        "Recall": rec,
        "Accuracy": acc,
        "F1": f1,
        "Samples": len(all_y_true_cls)
    }




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="path to preprocessed_1.csv")
    parser.add_argument("--models", nargs="+", required=True, help="one or more .pkl model paths")
    parser.add_argument("--neighbours", type=int, default=None, help="override neighbour_k at eval (optional)")
    args = parser.parse_args()

    set_seed()
    test = pd.read_csv(args.test)
    required = {"user_id","item_id", *FACTOR_COLS, TARGET_COL_R, TARGET_COL_C}
    miss = required - set(test.columns)
    if miss:
        raise ValueError(f"Missing columns in test: {miss}")

    rows = []
    for mpath in args.models:
        # res = evaluate_one_model(mpath, test, neighbour_k=args.neighbours)
        # rows.append([mpath, res["Samples"], res["RMSE"], res["Precision"], res["Recall"], res["Accuracy"], res["F1"]])
        model = load_model(mpath)
        # Content-based?
        if hasattr(model, "item_profiles") and not hasattr(model, "user_w_train"):
            res = evaluate_one_model_content_based(model, test, seed_mode="all")
        else:
            # 기존 user-based 평가
            res = evaluate_one_model(mpath, test)
        res = {"Model": os.path.basename(mpath), **res}
        rows.append(res)

    print()
    # print(tabulate(rows, headers=["Model_k_shrink_ridge","#Pairs","RMSE","Precision","Recall","Accuracy","F1"], floatfmt=".4f", tablefmt="github"))
    print(tabulate(rows, headers="keys", floatfmt=".4f", tablefmt="github"))
    print()

if __name__ == "__main__":
    main()