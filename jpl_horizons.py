"""
Живой парсер NASA JPL Horizons (REST API) для проекта «Вояджер» в системе Кербола.

Зачем: числа по реальному пролёту «Вояджера-2» у Юпитера (09.07.1979) в sim.py
зашиты в константы. Этот модуль позволяет получать их напрямую из JPL, строить
по ним фит v∞ из закона сохранения энергии (vis-viva) и сохранять снимок данных
в JSON — чтобы отчёт был воспроизводим, а числа не «менялись задним числом».

Запуск:  python3 jpl_horizons.py [--save out/jpl_snapshot.json] [--no-network]
Без сети скрипт читает ранее сохранённый снимок (или встроенный запасной вариант).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import urllib.parse
import urllib.request

import numpy as np

API = "https://ssd.jpl.nasa.gov/api/horizons.api"
MU_JUP = 1.26686534e8     # км^3/с^2
R_JUP = 71_492.0          # км

# --- идентификаторы тел в Horizons ---------------------------------------------------
VOYAGER2 = "-32"
JUPITER_BARYCENTER = "@599"  # центр Юпитера (префикс @ обязателен)
SUN = "500@10"               # Солнце (барицентр Солнечной системы)


def _plus_days(date_iso: str, days: int) -> str:
    d = dt.date.fromisoformat(date_iso) + dt.timedelta(days=days)
    return d.isoformat()


def _get(params: dict, timeout: int = 60) -> str:
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "VARKT-voyager/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_vectors(target: str, t0: str, t1: str, step: str, center: str) -> list[dict]:
    """Векторные эфемериды (положение + скорость) в эклиптике J2000. Единицы: км, км/с."""
    txt = _get({
        "format": "text", "COMMAND": f"'{target}'", "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'", "EPHEM_TYPE": "'VECTORS'", "CENTER": f"'{center}'",
        "START_TIME": f"'{t0}'", "STOP_TIME": f"'{t1}'", "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "'2'", "CSV_FORMAT": "'YES'", "TIME_DIGITS": "'SECONDS'",
    })
    rows, inside = [], False
    for line in txt.splitlines():
        if line.startswith("$$SOE"):
            inside = True
            continue
        if line.startswith("$$EOE"):
            break
        if inside:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                try:
                    rows.append(dict(jd=float(parts[0]), date=parts[1],
                                     r=[float(v) for v in parts[2:5]],
                                     v=[float(v) for v in parts[5:8]]))
                except ValueError:
                    continue
    return rows


def fit_vinf(rows: list[dict], r_lo_R: float = 20, r_hi_R: float = 120) -> dict:
    """
    Фит v∞ по закону сохранения энергии: v² = v∞² + 2μ/r.
    Линейная регрессия по x = 1/r даёт и v∞, и НАКЛОН 2μ — а значит, независимую
    проверку того, что данные «чистые» (сравниваем с табличным μ Юпитера).
    """
    rr = np.array([np.linalg.norm(r["r"]) for r in rows])
    vv = np.array([np.linalg.norm(r["v"]) for r in rows])
    m = (rr > r_lo_R * R_JUP) & (rr < r_hi_R * R_JUP)
    x, y = 1.0 / rr[m], vv[m] ** 2
    A = np.vstack([np.ones_like(x), x]).T
    (c0, c1), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([c0, c1])
    return dict(v_inf=float(math.sqrt(max(c0, 0.0))),
                mu_fit=float(c1 / 2),
                mu_true=MU_JUP,
                mu_rel_err=float(abs(c1 / 2 - MU_JUP) / MU_JUP),
                rms=float(np.sqrt(np.mean(resid ** 2))),
                n_points=int(m.sum()),
                r_range_R=[r_lo_R, r_hi_R])


def delta_at_radius(rp_km: float, vinf_km_s: float, r_km: float, mu: float = MU_JUP) -> float:
    """Угол между векторами скорости на входе и выходе на одном радиусе r (градусы)."""
    e = 1 + rp_km * vinf_km_s ** 2 / mu
    p = rp_km * (1 + e)
    cosnu = (p / r_km - 1) / e
    nu = math.acos(max(-1.0, min(1.0, cosnu)))
    s, c = math.sin(nu), math.cos(nu)
    return math.degrees(math.acos(max(-1.0, min(1.0, ((e + c) ** 2 - s ** 2) /
                                                 ((e + c) ** 2 + s ** 2)))))


def analyse_window(rows: list[dict], r_obs_R: float = 60.0) -> dict:
    """Перицентр, v∞, e, δ(формула), δ(эфемериды на равных радиусах)."""
    rr = np.array([np.linalg.norm(row["r"]) for row in rows])
    i_min = int(np.argmin(rr))
    rp = float(rr[i_min])
    fit = fit_vinf(rows)
    v_inf = fit["v_inf"]
    e = 1 + rp * v_inf ** 2 / MU_JUP

    # измеренный угол: ближайшие точки до/после перицентра на радиусе ≈ r_obs
    r_target = r_obs_R * R_JUP
    before = [row for i, row in enumerate(rows) if i < i_min]
    after = [row for i, row in enumerate(rows) if i > i_min]
    if not before or not after:
        return {}
    b = min(before, key=lambda row: abs(np.linalg.norm(row["r"]) - r_target))
    a = min(after, key=lambda row: abs(np.linalg.norm(row["r"]) - r_target))
    vb, va = np.array(b["v"]), np.array(a["v"])
    cosang = np.dot(vb, va) / np.linalg.norm(vb) / np.linalg.norm(va)
    delta_obs = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    r_b, r_a = np.linalg.norm(b["r"]), np.linalg.norm(a["r"])

    return dict(
        r_p_km=rp, r_p_R=rp / R_JUP, v_inf_km_s=v_inf,
        ecc=e, delta_asymptotic=math.degrees(2 * math.asin(1 / e)),
        b_km=rp * math.sqrt((e + 1) / (e - 1)),
        delta_observed=delta_obs,
        delta_model_at_r_before=delta_at_radius(rp, v_inf, r_b),
        delta_model_at_r_after=delta_at_radius(rp, v_inf, r_a),
        r_obs_before_R=r_b / R_JUP, r_obs_after_R=r_a / R_JUP,
        jd_pericenter=rows[i_min]["jd"], date_pericenter=rows[i_min]["date"],
        window=[rows[0]["date"], rows[-1]["date"]],
        fit=fit,
    )


def heliocentric_energy_change(t_before: str, t_after: str) -> dict:
    """Изменение гелиоцентрической скорости аппарата до и после пролёта."""
    # START и STOP не могут совпадать — берём окно в 1 сутки и первую точку
    v0 = fetch_vectors(VOYAGER2, t_before, _plus_days(t_before, 1), "1d", SUN)
    v1 = fetch_vectors(VOYAGER2, t_after, _plus_days(t_after, 1), "1d", SUN)
    if not v0 or not v1:
        return {}
    s0 = float(np.linalg.norm(v0[0]["v"]))
    s1 = float(np.linalg.norm(v1[0]["v"]))
    return dict(date_before=t_before, date_after=t_after,
                v_before_km_s=s0, v_after_km_s=s1, dv_km_s=s1 - s0)


def collect(quick: bool = False) -> dict:
    """Полный сбор: окно вокруг пролёта + энергетика."""
    step = "6h" if quick else "2h"
    rows = fetch_vectors(VOYAGER2, "1979-07-01", "1979-07-20", step, JUPITER_BARYCENTER)
    out = dict(
        source="NASA/JPL Horizons API (https://ssd.jpl.nasa.gov/api/horizons.api)",
        target="Voyager 2 (-32), Jupiter flyby",
        retrieved=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        step=step,
        flyby=analyse_window(rows),
        heliocentric=heliocentric_energy_change("1979-06-20", "1979-08-05"),
        raw_points=len(rows),
    )
    # набор точек сохраняем в сжатом виде — для перерисовки графиков без сети
    out["samples"] = [dict(jd=r["jd"], r=r["r"], v=r["v"]) for r in rows[:: max(1, len(rows) // 200)]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=os.path.join("out", "jpl_snapshot.json"))
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    if args.no_network:
        if not os.path.exists(args.save):
            print(f"Нет снимка {args.save} и сеть отключена", file=sys.stderr)
            return 1
        with open(args.save, encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            data = collect(quick=args.quick)
        except Exception as exc:  # сеть недоступна — откатываемся на снимок
            print(f"Сеть недоступна ({exc}); использую {args.save}", file=sys.stderr)
            with open(args.save, encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    fb = data["flyby"]
    print("=" * 78)
    print("«Вояджер-2» у Юпитера по данным JPL Horizons")
    print("=" * 78)
    print(f"  источник: {data['source']}")
    print(f"  снимок:   {data['retrieved']}, шаг {data['step']}, окно {fb['window']}")
    print(f"  перицентр r_p = {fb['r_p_km']:.0f} км = {fb['r_p_R']:.3f} R_Юпитера "
          f"({fb['date_pericenter'][:24]})")
    print(f"  фит vis-viva по кольцу {fb['fit']['r_range_R'][0]:.0f}–{fb['fit']['r_range_R'][1]:.0f} R_J "
          f"(N = {fb['fit']['n_points']}):")
    print(f"    v∞ = {fb['v_inf_km_s']:.3f} км/с;  2μ = {2*fb['fit']['mu_fit']:.4e} "
          f"(табличное {2*fb['fit']['mu_true']:.4e}; отн. откл. {fb['fit']['mu_rel_err']*100:.2f} %)")
    print(f"    СКО остатков по v² = {fb['fit']['rms']:.3e} (км/с)²")
    print(f"  e = 1 + r_p v∞²/μ = {fb['ecc']:.4f};  δ(асимптота) = {fb['delta_asymptotic']:.2f}°")
    print(f"  прицельный параметр b = {fb['b_km']:.0f} км = {fb['b_km']/R_JUP:.2f} R_J")
    print(f"  угол на равных радиусах {fb['r_obs_before_R']:.1f}/{fb['r_obs_after_R']:.1f} R_J:")
    print(f"    эфемериды  {fb['delta_observed']:.2f}°")
    print(f"    модель     {fb['delta_model_at_r_before']:.2f}°  "
          f"(расхождение {abs(fb['delta_model_at_r_before']-fb['delta_observed']):.2f}°)")
    hel = data.get("heliocentric", {})
    if hel:
        print(f"  гелиоцентрическая скорость: {hel['v_before_km_s']:.2f} → {hel['v_after_km_s']:.2f} км/с "
              f"(+{hel['dv_km_s']:.2f})")
    print(f"\nСнимок данных: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
