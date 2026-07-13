"""ML ile aday skorlama — gerçek ve paper mod ortak mantık."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional, Tuple

from ml.learner import ContinuousLearner


async def score_ml_candidates(
    bybit: Any,
    candidates: List[Dict[str, Any]],
    ml_signals: Any,
    build_ml_feature_row: Callable,
    tz: Any,
    *,
    learner: Optional[ContinuousLearner] = None,
    source: str = "real",
    long_min: float,
    short_max: float,
    validate_entry: Callable[[str, Dict[str, float]], Tuple[bool, str]],
) -> Tuple[List[Tuple[float, Dict[str, Any], str, float, List[float]]], List[Tuple[float, Dict[str, Any], str, float, Dict[str, float]]]]:
    """
    Dönüş:
      - scored: (confidence, coin, side, p_up, feats) — işlem adayları
      - observations: (p_up, coin, side, price, feat_dict) — öğrenme kaydı
    """
    sem_ml = asyncio.Semaphore(8)
    scored: List[Tuple[float, Dict[str, Any], str, float, List[float]]] = []
    observations: List[Tuple[float, Dict[str, Any], str, float, Dict[str, float]]] = []

    for coin in candidates:
        sym = coin.get("symbol")
        if not sym:
            continue
        feats = await build_ml_feature_row(bybit, sym, tz, sem_ml, ml_signals.feature_names)
        if feats is None:
            continue
        p_up = ml_signals.proba_up(feats)
        feat_dict = {name: float(val) for name, val in zip(ml_signals.feature_names, feats)}
        try:
            price = float(coin.get("lastPrice", coin.get("last_price")))
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue

        chg = coin.get("price_change_pct")
        chg_s = f" 5m {chg:+.2f}%" if chg is not None else ""

        if p_up >= long_min:
            ok, reason = validate_entry("long", feat_dict)
            if ok:
                scored.append((p_up, coin, "long", p_up, feats))
                observations.append((p_up, coin, "long", price, feat_dict))
                print(f"🤖 {sym}{chg_s} | ML P={p_up:.3f} → LONG ✓")
            else:
                print(f"⏭️ {sym}{chg_s} | LONG reddedildi: {reason}")
        elif p_up <= short_max:
            ok, reason = validate_entry("short", feat_dict)
            if ok:
                conf = 1.0 - p_up
                scored.append((conf, coin, "short", p_up, feats))
                observations.append((conf, coin, "short", price, feat_dict))
                print(f"🤖 {sym}{chg_s} | ML P={p_up:.3f} → SHORT ✓")
            else:
                print(f"⏭️ {sym}{chg_s} | SHORT reddedildi: {reason}")
        else:
            print(f"⏭️ {sym}{chg_s} | ML P={p_up:.3f} belirsiz, atlandı")

    if learner is not None and observations:
        for _conf, coin, side, price, feat_dict in learner.pick_observations(observations):
            sym = coin.get("symbol", "")
            p_up = ml_signals.proba_up(
                [feat_dict.get(n, 0.0) for n in ml_signals.feature_names]
            )
            learner.observe_candidate(
                symbol=sym,
                entry_price=price,
                features=feat_dict,
                p_up=p_up,
                source=source,
                side_hint=side,
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored, observations
