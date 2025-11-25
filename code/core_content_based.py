# content_based_core.py
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

RANDOM_SEED = 42
FACTOR_COLS = [
    "Class","Seat Comfort","Staff Service",
    "Food & Beverages","Inflight Entertainment","Value For Money"
]
TARGET_COL_R = "Overall Rating"     # 1..10
TARGET_COL_C = "Recommended"        # True/False


@dataclass
class ContentBasedModel:
    """Content-based item profiles with Bayesian smoothing in a scaled space.
       + Small calibration regression: r_hat = a*sim + b*seed + c*item_bias + d
       + Output clipping to [1,10]
       + Learned recommendation threshold (global, by maximizing F1)
    """
    alpha: float = 20.0                 # Empirical Bayes strength
    weight_mode: str = "rating"         # 'uniform' or 'rating'
    scaler: Optional[StandardScaler] = None
    mu_vec: Optional[np.ndarray] = None # global mean (scaled space)
    item_profiles: Dict[int, np.ndarray] = None
    item_counts: Dict[int, int] = None  # for reference/analysis

    # --- calibration regression params ---
    calib_coef: Optional[np.ndarray] = None   # [a, b, c]
    calib_intercept: float = 0.0              # d
    # item 평균 평점 (회귀 입력 c*item_bias에서 사용)
    item_bias: Optional[Dict[int, float]] = None

    # clipping 범위
    clip_min: float = 1.0
    clip_max: float = 10.0

    # --- recommendation decision (global threshold) ---
    rec_decision: str = "sim"       # "sim" 또는 "rating"
    tau_sim: float = 0.5            # sim 임계값
    tau_rating: float = 5.0         # r_hat 임계값

    # --- NEW: seed Recommended별 임계값 (pattern) ---
    tau_sim_pos: float = 0.5        # seed가 Recommended=1일 때 패턴
    tau_sim_neg: float = 0.5        # seed가 Recommended=0일 때 패턴
    tau_rating_pos: float = 5.0     # rating 모드에서 seed=1
    tau_rating_neg: float = 5.0     # rating 모드에서 seed=0

    # =========================
    # Building common space
    # =========================
    def fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit a StandardScaler on ALL raw factor rows to define a common space."""
        X = df[FACTOR_COLS].to_numpy(dtype=float)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(X)

    def fit_calibration(self, X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> None:
        """
        작은 회귀를 학습해서 (a, b, c, d)를 세팅한다.
        X shape = (n_samples, 3)  where columns = [sim, seed_rating, item_bias]
        y shape = (n_samples,)
        """
        if X.size == 0 or y.size == 0:
            # 데이터가 없으면 기본값 (스케일형 베이스라인) 유지
            self.calib_coef = np.array([1.0, 1.0, 0.0], dtype=float)
            self.calib_intercept = 0.0
            return

        model = Ridge(alpha=alpha, fit_intercept=True, random_state=RANDOM_SEED)
        model.fit(X, y)
        coef = model.coef_.astype(float)
        if coef.shape[0] != 3:
            coef = np.pad(coef[:3], (0, max(0, 3 - coef.shape[0])), constant_values=0.0)

        self.calib_coef = coef
        self.calib_intercept = float(model.intercept_)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X) if self.scaler is not None else X

    # =========================
    # Item profiles (scaled + EB)
    # =========================
    def build_item_profiles(self, df: pd.DataFrame) -> None:
        """
        Build item profiles in the SCALED space:
        1) transform raw factors
        2) compute weighted mean per item (uniform or rating-weighted)
        3) empirical Bayes smoothing: x_i <- (n/(n+alpha))*m_i + (alpha/(n+alpha))*mu
        """
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler(df) before build_item_profiles(df).")

        # 1) 모든 행을 스케일
        X_scaled = self._transform(df[FACTOR_COLS].to_numpy(dtype=float))
        ratings  = df[TARGET_COL_R].to_numpy(dtype=float)
        items    = df["item_id"].to_numpy(dtype=int)

        # 2) 전역 평균(mu) — 스케일된 공간에서 계산 (prior)
        self.mu_vec = X_scaled.mean(axis=0).astype(float)

        # 3) 아이템별 가중 평균
        sum_wx = defaultdict(lambda: np.zeros(X_scaled.shape[1], dtype=float))
        sum_w  = Counter()

        if self.weight_mode == "rating":
            # 평점 기반 가중
            w = ratings.copy()
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            w[w < 0.0] = 0.0
        else:
            w = np.ones_like(ratings, dtype=float)

        for x, iid, wi in tqdm(zip(X_scaled, items, w),
                               total=len(items), desc="Build weighted means"):
            if wi <= 0.0:
                continue
            sum_wx[iid] += wi * x
            sum_w[iid]  += wi

        item_mean = {}
        item_n    = {}
        all_item_ids = df["item_id"].to_numpy()
        for iid in sum_w.keys():
            if sum_w[iid] > 0:
                m_i = sum_wx[iid] / sum_w[iid]
                item_mean[iid] = m_i
                item_n[iid] = int((all_item_ids == iid).sum())
            else:
                continue

        # 4) Empirical Bayes smoothing in scaled space
        profiles = {}
        a = float(self.alpha)
        for iid, m_i in item_mean.items():
            n_i = item_n[iid]
            prof = (n_i/(n_i+a))*m_i + (a/(n_i+a))*self.mu_vec
            profiles[iid] = prof.astype(float)

        self.item_profiles = profiles
        self.item_counts   = item_n

    # =========================
    # Bias (mean ratings)
    # =========================
    def build_item_bias(self, df: pd.DataFrame) -> None:
        """아이템 평균 평점 (회귀 입력 feature로 사용)."""
        self.item_bias = df.groupby("item_id")[TARGET_COL_R].mean().to_dict()

    # =========================
    # Pair features for calibration
    # =========================
    def _cos(self, u: np.ndarray, v: np.ndarray) -> float:
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0: return 0.0
        return float(np.dot(u, v) / (nu*nv))

    def make_pair_features(self, df: pd.DataFrame):
        """
        작은 회귀 학습용 (X, y) 생성.
        - X = [sim(seed,target), seed_rating, item_bias(target)]
        - y = target_rating
        같은 유저의 (seed → target) 모든 쌍을 생성 (seed != target).
        """
        if self.item_profiles is None or self.item_bias is None:
            raise RuntimeError("Call build_item_profiles(...) and build_item_bias(...) first.")

        X_list, y_list = [], []
        for _, g in tqdm(df.groupby("user_id"), desc="Build (seed,target) pairs"):
            items = g["item_id"].to_numpy(dtype=int)
            ratings = g[TARGET_COL_R].to_numpy(dtype=float)
            r_by_item = {int(i): float(r) for i, r in zip(items, ratings)}

            for seed in items:
                x_s = self.item_profiles.get(int(seed))
                if x_s is None:
                    continue
                seed_rating = r_by_item[int(seed)]
                for target in items:
                    if target == seed:
                        continue
                    x_i = self.item_profiles.get(int(target))
                    if x_i is None:
                        continue
                    sim = self._cos(x_s, x_i)
                    ib  = float(self.item_bias.get(int(target), np.nan))
                    if np.isnan(ib):
                        continue
                    y   = r_by_item[int(target)]
                    X_list.append([sim, seed_rating, ib])
                    y_list.append(y)

        if not X_list:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=float)

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)
        return X, y

    # =========================
    # Predict with calibration
    # =========================
    def predict_calibrated(self, sim: float, seed_rating: float, item_id: int) -> float:
        """
        r_hat = a*sim + b*seed_rating + c*item_bias(item_id) + d
        + clip to [clip_min, clip_max]
        """
        if self.calib_coef is None or self.item_bias is None:
            r = max(sim, 0.0) * float(seed_rating)
            return float(np.clip(r, self.clip_min, self.clip_max))

        a, b, c = self.calib_coef.tolist()
        d = float(self.calib_intercept)
        ib = float(self.item_bias.get(int(item_id), (self.clip_min + self.clip_max)/2.0))
        r = a*float(sim) + b*float(seed_rating) + c*ib + d
        return float(np.clip(r, self.clip_min, self.clip_max))

    # ---- Helper: user prototype from seeds (optional use) ----
    def user_vector_from_seeds(self, seed_item_ids):
        vecs = []
        for iid in seed_item_ids:
            x = self.item_profiles.get(int(iid))
            if x is not None:
                vecs.append(x)
        if not vecs:
            return None
        return np.mean(np.vstack(vecs), axis=0)

    # ---- Similarity (cosine) ----
    @staticmethod
    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0.0 or nv == 0.0: return 0.0
        return float(np.dot(u, v) / (nu*nv))

    # =========================
    # NEW: Learn global threshold for Recommended
    # =========================
    def _f1_from_preds(self, y_true: List[int], y_pred: List[int]) -> Tuple[float,float,float,float]:
        tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
        fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
        fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1   = (2*prec*rec) / max(prec+rec, 1e-12)
        return f1, prec, rec, tp+fp+fn+sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)

    def fit_rec_threshold(self, df: pd.DataFrame, mode: str = "sim",
                        grid: Optional[np.ndarray] = None) -> Tuple[float, dict]:
        """
        seed→target 쌍에서 점수(score)와 라벨(y_rec)을 만들고,
        F1이 최대가 되는 임계값 τ를 학습한다.

        + (2) 각 사용자가 평가한 item 중 하나(seed)의 Recommended를 확인하고,
             나머지 target 아이템들에 대한 유사도와 Recommended 사이의 관계를 학습.
        + (4) seed가 Recommended=1인 경우와 0인 경우를 분리해서
             서로 다른 유사도 임계값 패턴(tau_pos, tau_neg)을 각각 학습.

          - mode="sim":    score = cosine(item_profile[seed], item_profile[target])
          - mode="rating": score = predict_calibrated(sim, seed_rating, target_item)

        반환: (global_tau, {f1, precision, recall, n, tau_pos, tau_neg, stats_pos, stats_neg})
        """
        if self.item_profiles is None:
            raise RuntimeError("Call build_item_profiles(...) first.")
        if mode not in ("sim", "rating"):
            raise ValueError("mode must be 'sim' or 'rating'.")

        # (user_id, item_id) -> Recommended(0/1)
        rec_label = {
            (int(r.user_id), int(r.item_id)):
                1 if r[TARGET_COL_C] in (True, 1, "True", "true") else 0
            for _, r in df.iterrows()
        }

        # 전체 패턴용
        scores_all: List[float] = []
        labels_all: List[int]   = []

        # NEW: seed Recommended=1 / 0 별로 나눠서 패턴 수집
        scores_pos: List[float] = []   # seed_rec = 1
        labels_pos: List[int]   = []
        scores_neg: List[float] = []   # seed_rec = 0
        labels_neg: List[int]   = []

        for _, g in tqdm(df.groupby("user_id"), desc=f"Build pairs for τ ({mode})"):
            items = g["item_id"].to_numpy(dtype=int)
            ratings = g[TARGET_COL_R].to_numpy(dtype=float)
            r_by_item = {int(i): float(r) for i, r in zip(items, ratings)}

            uid = int(g["user_id"].iloc[0])
            # 해당 유저가 각 아이템을 추천/비추천 했는지
            rec_by_item = {
                int(row.item_id): (
                    1 if row[TARGET_COL_C] in (True, 1, "True", "true") else 0
                )
                for _, row in g.iterrows()
            }

            for seed in items:
                x_s = self.item_profiles.get(int(seed))
                if x_s is None:
                    continue
                seed_rating = r_by_item[int(seed)]
                seed_rec = rec_by_item[int(seed)]  # NEW: seed의 Recommended 확인

                for target in items:
                    if target == seed:
                        continue
                    x_i = self.item_profiles.get(int(target))
                    if x_i is None:
                        continue

                    sim = self._cos(x_s, x_i)
                    if mode == "sim":
                        score = sim
                    else:
                        score = self.predict_calibrated(sim, seed_rating, int(target))

                    # target의 Recommended 라벨
                    y = rec_label[(uid, int(target))]

                    # 전체 패턴
                    scores_all.append(float(score))
                    labels_all.append(int(y))

                    # seed_rec에 따라 양쪽 패턴으로 분리
                    if seed_rec == 1:
                        scores_pos.append(float(score))
                        labels_pos.append(int(y))
                    else:
                        scores_neg.append(float(score))
                        labels_neg.append(int(y))

        if not scores_all:
            # 데이터 부족 시 기본값 유지
            default_tau = self.tau_sim if mode == "sim" else self.tau_rating
            return default_tau, {
                "f1": 0.0, "precision": 0.0, "recall": 0.0, "n": 0,
                "tau_pos": None, "tau_neg": None,
                "stats_pos": None, "stats_neg": None,
            }

        scores_all_np = np.asarray(scores_all, dtype=float)
        labels_all_np = np.asarray(labels_all, dtype=int)

        # grid 설정
        if grid is None:
            if mode == "sim":
                grid = np.linspace(0.0, 1.0, 201)
            else:
                grid = np.linspace(self.clip_min, self.clip_max, 181)

        # 내부 헬퍼: 주어진 score/label에 대해 best τ 찾기
        def _search_best_tau(scores_np: np.ndarray,
                             labels_np: np.ndarray,
                             grid_vals: np.ndarray):
            best = (-1.0, 0.0, 0.0, 0.0, None)  # (f1, prec, rec, n, tau)
            labels_list = labels_np.tolist()
            for tau in grid_vals:
                y_pred = (scores_np >= tau).astype(int).tolist()
                f1, prec, rec, n = self._f1_from_preds(labels_list, y_pred)
                if f1 > best[0]:
                    best = (f1, prec, rec, n, float(tau))
            return best

        # 1) 전체 데이터에 대한 전역 임계값 (기존 기능)
        best_all = _search_best_tau(scores_all_np, labels_all_np, grid)

        # 2) NEW: seed_rec = 1 / 0 에 대한 패턴별 임계값
        best_pos = best_neg = None
        if scores_pos:
            best_pos = _search_best_tau(
                np.asarray(scores_pos, dtype=float),
                np.asarray(labels_pos, dtype=int),
                grid
            )
        if scores_neg:
            best_neg = _search_best_tau(
                np.asarray(scores_neg, dtype=float),
                np.asarray(labels_neg, dtype=int),
                grid
            )

        # 전역 τ는 기존과 동일하게 세팅
        tau_all = best_all[4]
        if mode == "sim":
            self.tau_sim = tau_all
            self.rec_decision = "sim"
            # NEW: seed_rec별 패턴 임계값 저장
            if best_pos is not None:
                self.tau_sim_pos = best_pos[4]
            if best_neg is not None:
                self.tau_sim_neg = best_neg[4]
        else:
            self.tau_rating = tau_all
            self.rec_decision = "rating"
            if best_pos is not None:
                self.tau_rating_pos = best_pos[4]
            if best_neg is not None:
                self.tau_rating_neg = best_neg[4]

        # 리포트용 정보 (전역 + 패턴별)
        report = {
            "f1":       best_all[0],
            "precision": best_all[1],
            "recall":    best_all[2],
            "n":         int(best_all[3]),
            "tau_pos":   best_pos[4] if best_pos is not None else None,
            "tau_neg":   best_neg[4] if best_neg is not None else None,
            "stats_pos": {
                "f1": best_pos[0], "precision": best_pos[1],
                "recall": best_pos[2], "n": int(best_pos[3])
            } if best_pos is not None else None,
            "stats_neg": {
                "f1": best_neg[0], "precision": best_neg[1],
                "recall": best_neg[2], "n": int(best_neg[3])
            } if best_neg is not None else None,
        }

        return tau_all, report


    # =========================
    # NEW: Predict Recommended using learned τ
    # =========================
    def predict_rec(self, sim: float, seed_rating: float, item_id: int) -> int:
        """
        학습된 전역 임계값(τ)에 따라 Recommended(0/1) 예측.
          - rec_decision == "sim":     sim >= tau_sim
          - rec_decision == "rating":  r_hat(sim,seed,item_id) >= tau_rating
        """
        if self.rec_decision == "rating":
            score = self.predict_calibrated(sim, seed_rating, item_id)
            return 1 if score >= self.tau_rating else 0
        else:
            # 기본은 유사도 기반
            return 1 if sim >= self.tau_sim else 0