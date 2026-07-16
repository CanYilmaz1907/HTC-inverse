import asyncio
import datetime as dt

from bybit_client import BybitClient
from config import load_config
from scanner import _get_last_5m_change_pct, _get_latest_actual_funding_rate


async def main() -> None:
    cfg = load_config()
    client = BybitClient(cfg.bybit)
    tz = dt.timezone.utc

    instruments = await client.get_instruments_info()
    perpetual = []
    for inst in instruments:
        if inst.get("status") != "Trading":
            continue
        if str(inst.get("contractType") or "").lower() not in {"perpetual", "linearperpetual"}:
            continue
        sym = inst.get("symbol")
        if not sym:
            continue
        try:
            interval = int(inst.get("fundingInterval"))
        except Exception:
            interval = None
        perpetual.append((sym, interval))

    allowed = set(cfg.criteria.allowed_funding_intervals_min)
    allowed_syms = [s for s, interval in perpetual if interval in allowed]
    print("perpetual_total:", len(perpetual))
    print("allowed_interval_total:", len(allowed_syms))

    sem = asyncio.Semaphore(10)
    # sample first 30 symbols to see if funding history exists
    sample = allowed_syms[:30]
    fr_tasks = [_get_latest_actual_funding_rate(client, s, sem) for s in sample]
    chg_tasks = [_get_last_5m_change_pct(client, s, sem) for s in sample]
    frs = await asyncio.gather(*fr_tasks, return_exceptions=True)
    chgs = await asyncio.gather(*chg_tasks, return_exceptions=True)

    neg_fr = 0
    has_fr = 0
    big_move = 0
    for s, fr, chg in zip(sample, frs, chgs):
        if not isinstance(fr, Exception) and fr is not None:
            has_fr += 1
            if fr < 0:
                neg_fr += 1
        if not isinstance(chg, Exception) and chg is not None and chg >= cfg.criteria.min_price_change_percent:
            big_move += 1

    print("sample_size:", len(sample))
    print("sample_funding_history_present:", has_fr)
    print("sample_actual_funding_negative:", neg_fr)
    print("sample_5m_move_ge_threshold:", big_move, f"(threshold={cfg.criteria.min_price_change_percent}%)")

    # show a few rows
    for s, fr, chg in list(zip(sample, frs, chgs))[:10]:
        fr_v = None if isinstance(fr, Exception) else fr
        chg_v = None if isinstance(chg, Exception) else chg
        print(s, "funding=", fr_v, "chg5m%=", chg_v)


if __name__ == "__main__":
    asyncio.run(main())

