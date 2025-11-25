# hybrid_user_cf_core_noprofile.py
# -*- coding: utf-8 -*-
import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from tqdm.auto import tqdm

RANDOM_SEED = 42
FACTOR_COLS = [
    "Class","Seat Comfort","Staff Service",
    "Food & Beverages","Inflight Entertainment","Value For Money"
]
TARGET_COL_R = "Overall Rating"
TARGET_COL_C = "Recommended"

def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v/(n if n>0 else 1.0)

def _shrink(n: int, alpha: float) -> float:
    return n/(n+alpha)

@dataclass
class HybridUserCFNoProfile:
    """Pure user-based: no item-profile in train (only raw factor rows)."""
    neighbour_k: int = 100
    shrink_alpha: float = 10.0
    ridge_alpha: float = 1.0
    content_weight: float = 0.0  # γ=0 → 순수 user-based

    # learned params
    scaler: Optional[StandardScaler] = None
    user_w_train: Dict[int, np.ndarray] = None
    user_b_train: Dict[int, float] = None
    user_n_train: Dict[int, int] = None
    global_w: Optional[np.ndarray] = None
    global_b: float = 0.0
    clf: Optional[LogisticRegression] = None  # Recommended classifier

    # -------------------- Fit scaler on raw factors --------------------
    def fit_scaler_on_factors(self, df: pd.DataFrame):
        X = df[FACTOR_COLS].to_numpy(dtype=float)
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.scaler.fit(X)

    def _tx(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X) if self.scaler is not None else X

    # -------------------- Train user preferences (no item-profile) --------------------
    def fit_user_preferences_train(self, df: pd.DataFrame):
        """
        Per-user Ridge: y = b_u + w_u · (scaled raw factor vector of that row).
        """
        self.global_b = float(df[TARGET_COL_R].mean()) if len(df) else 0.0

        user_w, user_b, user_n = {}, {}, {}
        Xw_all, yw_all = [], []

        for uid, g in tqdm(df.groupby("user_id"), desc="Train: user preferences", total=df["user_id"].nunique()):
            X = self._tx(g[FACTOR_COLS].to_numpy(dtype=float))
            y = g[TARGET_COL_R].to_numpy(dtype=float)

            if len(y) >= 2:
                m = Ridge(alpha=self.ridge_alpha, fit_intercept=True, random_state=RANDOM_SEED)
                m.fit(X, y)
                w = m.coef_.astype(float); b = float(m.intercept_)
            elif len(y) == 1:
                w = np.zeros(len(FACTOR_COLS), dtype=float); b = float(y[0])
            else:
                w = np.zeros(len(FACTOR_COLS), dtype=float); b = 0.0

            user_w[uid], user_b[uid], user_n[uid] = w, b, int(len(y))
            if len(y) >= 2:
                Xw_all.append(X); yw_all.append(y)

        if Xw_all:
            Xgl = np.vstack(Xw_all); ygl = np.concatenate(yw_all)
            m = Ridge(alpha=self.ridge_alpha, fit_intercept=True, random_state=RANDOM_SEED)
            m.fit(Xgl, ygl)
            self.global_w = m.coef_.astype(float)
        else:
            self.global_w = np.zeros(len(FACTOR_COLS), dtype=float)

        self.user_w_train, self.user_b_train, self.user_n_train = user_w, user_b, user_n

    # -------------------- K-fold features for Recommended (Δ/ρ) --------------------
    def _build_fold_sums(self, df_fold: pd.DataFrame):
        """Global sums/counts and per-(u,i) raw vectors for LOUO within a fold."""
        sums = defaultdict(lambda: np.zeros(len(FACTOR_COLS), dtype=float))
        cnts = Counter()
        ui_vec = {}

        for _, r in df_fold.iterrows():
            uid, iid = int(r["user_id"]), int(r["item_id"])
            v = r[FACTOR_COLS].to_numpy(dtype=float)
            sums[iid] += v; cnts[iid] += 1
            ui_vec[(uid, iid)] = v
        return sums, cnts, ui_vec

    def _louo_vec(self, iid: int, uid: int,
                  sums: Dict[int, np.ndarray], cnts: Dict[int, int],
                  ui_vec: Dict[Tuple[int,int], np.ndarray]) -> Optional[np.ndarray]:
        if iid not in cnts or cnts[iid] <= 0: return None
        s = sums[iid].copy(); n = cnts[iid]
        key = (uid, iid)
        if key in ui_vec:
            s -= ui_vec[key]; n -= 1
        if n <= 0: return None
        x = (s / n).reshape(1, -1)  # fold-내 LOUO 평균 (raw factors)
        return self._tx(x).ravel()  # train 스케일러로 정규화

    def train_classifier_with_kfold(self, df: pd.DataFrame, n_splits: int = 3):
        """
        Δ/ρ는 fold-내 user-user 이웃의 (동일 아이템에 대한) 추천율/잔차로 구성.
        베이스라인 r0 = b_u + γ*(w_u·x_i) (여기서 γ=0이므로 사실상 b_u)
        """
        γ = float(self.content_weight)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        df = df.reset_index(drop=True)

        Xf_list, Y_list = [], []

        for fold_idx, (_, val_idx) in enumerate(kf.split(df), start=1):
            df_fold = df.iloc[val_idx].copy()

            # fold-내 LOUO용 준비
            sums, cnts, ui_vec = self._build_fold_sums(df_fold) #item에 대한 profile별 점수 누적, item 가짓수, uid,iid 쌍의 profile별 점수

            # fold-내 user별 w,b (raw factor → scaled)
            user_w_f, user_b_f, user_n_f = {}, {}, {}
            for uid, g in tqdm(df_fold.groupby("user_id"), leave=False, desc=f"Fold{fold_idx}: fit w,b"):
                X = self._tx(g[FACTOR_COLS].to_numpy(dtype=float))
                y = g[TARGET_COL_R].to_numpy(dtype=float)
                if len(y) >= 2:
                    m = Ridge(alpha=self.ridge_alpha, fit_intercept=True, random_state=RANDOM_SEED)
                    m.fit(X, y)
                    w = m.coef_.astype(float); b = float(m.intercept_)
                elif len(y) == 1:
                    w = np.zeros(len(FACTOR_COLS), dtype=float); b = float(y[0])
                else:
                    w = np.zeros(len(FACTOR_COLS), dtype=float); b = 0.0
                user_w_f[uid], user_b_f[uid], user_n_f[uid] = w, b, int(len(y))

            Wn = {u: _norm(w) for u, w in user_w_f.items()}
            rec_label = {(int(r.user_id), int(r.item_id)):
                         1 if r[TARGET_COL_C] in (True,1,"True","true") else 0
                         for _, r in df_fold.iterrows()}
            rating_label = {(int(r.user_id), int(r.item_id)): float(r[TARGET_COL_R])
                            for _, r in df_fold.iterrows()}
            user_items = defaultdict(list)
            for _, r in df_fold.iterrows():
                user_items[int(r.user_id)].append(int(r.item_id))

            # 각 사용자에 대해 seed 하나 선택 → 나머지로 Δ/ρ 생성
            for uid, items in tqdm(user_items.items(), leave=False, desc=f"Fold{fold_idx}: rec feats"):
                if len(items) < 2: 
                    continue

                # seed 선택 및 1-shot 약 업데이트
                rnd = np.random.default_rng(RANDOM_SEED + fold_idx + uid)
                seed = rnd.choice(items)
                y_seed = rating_label[(uid, seed)]
                x_seed = self._louo_vec(seed, uid, sums, cnts, ui_vec)

                w_prior = self.global_w.copy() if self.global_w is not None else np.zeros(len(FACTOR_COLS))
                b_prior = self.global_b
                if x_seed is not None:
                    denom = self.ridge_alpha + float(x_seed.dot(x_seed))
                    adj = (y_seed - b_prior - float(w_prior.dot(x_seed))) / denom
                    w_u = w_prior + adj * x_seed
                    b_u = (b_prior*self.ridge_alpha + y_seed) / (self.ridge_alpha + 1.0)
                else:
                    w_u, b_u = w_prior, b_prior

                # 이웃(top-K, shrink)
                neigh = []
                for v, wv in Wn.items():
                    if v == uid: 
                        continue
                    s = float(w_u.dot(wv))
                    if s <= 0: # 유사도가 0 이하인 사용자 제외
                        continue
                    s *= _shrink(user_n_f.get(v,0), self.shrink_alpha) * _shrink(1, self.shrink_alpha) # 나머지 사용자에 대해 이웃 숫자로 shrink 설정
                    if s > 0: 
                        neigh.append((v, s)) #userId와 shrink 반영된 유사도 저장
                neigh.sort(key=lambda x: -x[1])
                neigh = neigh[: self.neighbour_k]

                # seed 제외 나머지 아이템들에서 샘플 생성
                for iid in items:
                    if iid == seed:
                        continue
                    # 콘텐츠 베이스(γ=0이면 b_u)
                    x_i = self._louo_vec(iid, uid, sums, cnts, ui_vec)
                    base = float(w_u.dot(x_i)) if (x_i is not None) else 0.0
                    r0 = b_u + γ * base

                    # CF 특징(Δ/ρ): 이웃들이 같은 iid를 본 경우에만
                    num, den = 0.0, 0.0
                    num_res, den_res = 0.0, 0.0
                    for v, s in neigh: # 이웃 id, 조정된 유사도
                        if (v, iid) in rating_label: # 이웃 v가 iid에 내린 평가로 rating과 recommended 각각의 가중평균 계산
                            den += abs(s)
                            num += s * (1 if rec_label[(v,iid)]==1 else 0)
                            # γ=0 → r0_v ≈ b_v (x_i 불필요)
                            r0_v = user_b_f.get(v, 0.0)
                            num_res += s * (rating_label[(v,iid)] - r0_v)
                            den_res += abs(s)
                    rho = (num/den) if den>0 else 0.0
                    delta = (num_res/den_res) if den_res>0 else 0.0

                    y_bin = rec_label[(uid, iid)]
                    # item bias는 사용하지 않으므로 0.0
                    Xf_list.append([r0, delta, rho, b_u, 0.0])
                    Y_list.append(y_bin)

        Xf = np.asarray(Xf_list, dtype=float)
        Y  = np.asarray(Y_list, dtype=int)
        if len(Y) == 0:
            self.clf = None
            return
        self.clf = LogisticRegression(
            max_iter=300, solver="lbfgs", class_weight="balanced", random_state=RANDOM_SEED
        )
        self.clf.fit(Xf, Y)