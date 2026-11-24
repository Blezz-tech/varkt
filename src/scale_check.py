"""
Оценка точности модели точечных сфер действия (patched conics) — 4-й уровень верификации.

Главное допущение модели: вдали от Юла аппарат движется только в поле Кербола,
а внутри сферы действия — только в поле Юла. Реальная динамика — задача трёх тел.
Здесь две численные модели запускаются на ОДНИХ И ТЕХ ЖЕ начальных условиях:

  1) patched conics — модель из sim.py;
  2) полная задача трёх тел (Солнце + Юл на круговой орбите + аппарат).

Сравниваются: перицентр, угол поворота вектора скорости относительно Юла, афелий.

Важно про единицы: все расстояния — метры, скорости — м/с, время — секунды.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import (MU_SUN, MU_JOOL, R_JOOL, A_JOOL, SOI_JOOL, DAY,
                 launch_window, delta_hyp, b_hyp, helio_elements, post_flyby, OUT)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True, "savefig.bbox": "tight"})

OMEGA_JOOL = math.sqrt(MU_SUN / A_JOOL ** 3)      # угловая скорость Юла по орбите
V_JOOL = math.sqrt(MU_SUN / A_JOOL)


def sun_and_jool(pos, t):
    """Положения и скорости Солнца (в 0) и Юла (круговая орбита) в момент t."""
    rj = np.array([A_JOOL * math.cos(OMEGA_JOOL * t), A_JOOL * math.sin(OMEGA_JOOL * t)])
    vj = np.array([-A_JOOL * OMEGA_JOOL * math.sin(OMEGA_JOOL * t),
                   A_JOOL * OMEGA_JOOL * math.cos(OMEGA_JOOL * t)])
    return np.zeros(2), rj, vj


def accel(pos, t, mu_sun=MU_SUN, mu_p=MU_JOOL):
    """Суммарное ускорение от Солнца и Юла."""
    _, rj, _ = sun_and_jool(pos, t)
    d_sun = -mu_sun * pos / np.dot(pos, pos) ** 1.5
    rel = pos - rj
    d_jool = -mu_p * rel / np.dot(rel, rel) ** 1.5
    return d_sun + d_jool


def initial_state(v_inf: float, rp: float, side: int = +1, frac_soi: float = 0.95):
    """
    Старт на границе сферы действия Юла.

    Геометрия (в системе Юла): входящая асимптота направлена вдоль +x,
    прицельный параметр отложен по y. В гелиоцентрической системе поворачиваем
    так, чтобы относительная скорость была направлена против орбитального
    движения Юла (аппарат догоняет планету сзади — как при гоманновском перелёте).
    """
    b = b_hyp(rp, v_inf)
    r_chk = frac_soi * SOI_JOOL
    if b >= r_chk:
        raise ValueError("прицельный параметр больше сферы действия")
    # локальные координаты (относительно Юла, асимптота вдоль +x)
    r_loc = np.array([-math.sqrt(r_chk ** 2 - b ** 2), side * b])
    v_loc = np.array([v_inf, 0.0])
    # поворот на -90°: +x -> -y (входящая скорость направлена против движения Юла)
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])
    r_off, v_rel = R @ r_loc, R @ v_loc

    _, rj, vj = sun_and_jool(None, 0.0)
    return rj + r_off, vj + v_rel, r_chk


def integrate_3body(v_inf: float, rp: float, side: int = +1, dt: float = 60.0):
    """Полная задача трёх тел: RK4, постоянный шаг, до возврата на радиус старта."""
    pos, vel, r_check = initial_state(v_inf, rp, side)
    t = 0.0
    r_min, t_min = np.linalg.norm(pos - sun_and_jool(pos, t)[1]), 0.0
    hist = [(pos.copy(), vel.copy())]
    v_rel_0 = vel - sun_and_jool(pos, t)[2]

    for _ in range(3_000_000):
        k1v = accel(pos, t);                 k1r = vel
        k2v = accel(pos + 0.5 * dt * k1r, t + 0.5 * dt); k2r = vel + 0.5 * dt * k1v
        k3v = accel(pos + 0.5 * dt * k2r, t + 0.5 * dt); k3r = vel + 0.5 * dt * k2v
        k4v = accel(pos + dt * k3r, t + dt); k4r = vel + dt * k3v
        pos = pos + dt / 6 * (k1r + 2 * k2r + 2 * k3r + k4r)
        vel = vel + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt

        _, rj, vj = sun_and_jool(pos, t)
        rel = pos - rj
        d = np.linalg.norm(rel)
        if d < r_min:
            r_min, t_min = d, t
        if len(hist) == 0 or np.linalg.norm(pos - hist[-1][0]) > 0.01 * np.linalg.norm(pos):
            hist.append((pos.copy(), vel.copy()))
        # вышли обратно на радиус старта (после перицентра) — конец пролёта
        if d > r_check and t > DAY:
            break
    hist = np.array([[p[0], p[1], v[0], v[1]] for p, v in hist])
    pos, vel = hist[-1, :2], hist[-1, 2:]
    _, rj, vj = sun_and_jool(pos, t)
    v_rel_1 = vel - vj
    delta = math.degrees(math.acos(np.clip(
        np.dot(v_rel_0, v_rel_1) / np.linalg.norm(v_rel_0) / np.linalg.norm(v_rel_1), -1, 1)))
    el = helio_elements(pos, vel)
    return dict(r_min=r_min, d_end=d, delta=delta, apo=el["r_apo"], e=el["e"], t=t,
                hist=hist, r_min_pc=rp)


def compare(tr, h_km: float, side: int = +1, dt: float = 60.0):
    rp = R_JOOL + h_km * 1e3
    res = integrate_3body(tr.v_inf_j, rp, side=side, dt=dt)
    vinf_out, v_out = post_flyby(rp, tr.v_inf_j, side, tr.v_jool)
    el_pc = helio_elements(np.array([A_JOOL, 0.0]), v_out)
    return dict(h_km=h_km,
                delta_pc=math.degrees(delta_hyp(rp, tr.v_inf_j)),
                delta_3body=res["delta"],
                rmin_pc_km=rp / 1e3, rmin_3body_km=res["r_min"] / 1e3,
                t_days=res["t"] / DAY,
                apo_pc=el_pc["r_apo"] / 1e9, apo_3body=res["apo"] / 1e9,
                e_pc=el_pc["e"], e_3body=res["e"])


def main():
    tr = launch_window()
    print("=" * 84)
    print("СФЕРЫ ДЕЙСТВИЯ vs ЗАДАЧА ТРЁХ ТЕЛ")
    print("=" * 84)
    rows = []
    for h in (20_000, 50_000, 100_000, 500_000):
        r = compare(tr, h)
        rows.append(r)
        print(f"  h_p = {h:7d} км:")
        print(f"    δ (отн. Юла)  : сферы {r['delta_pc']:7.2f}°  |  3 тела {r['delta_3body']:7.2f}°"
              f"   Δ = {r['delta_3body']-r['delta_pc']:+7.2f}°")
        print(f"    перицентр     : сферы {r['rmin_pc_km']:9.0f} км | 3 тела {r['rmin_3body_km']:9.0f} км")
        print(f"    афелий        : сферы {r['apo_pc']:9.2f} | 3 тела {r['apo_3body']:9.2f} млн км")
    with open(os.path.join(OUT, "conics_vs_3body.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\n  Сходимость по шагу интегрирования (h_p = 100 000 км):")
    conv = []
    for dt in (240, 120, 60, 30):
        res = integrate_3body(tr.v_inf_j, R_JOOL + 100_000e3, dt=float(dt))
        conv.append((dt, res["apo"] / 1e9, res["delta"]))
        print(f"    шаг {dt:4.0f} c → афелий {res['apo']/1e9:8.4f} млн км, δ = {res['delta']:7.3f}°")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))
    hs = [r["h_km"] for r in rows]
    ax = axes[0]
    ax.plot(hs, [r["delta_pc"] for r in rows], "o-", lw=2, ms=8, color="#1a73e8",
            label="модель сфер действия")
    ax.plot(hs, [r["delta_3body"] for r in rows], "s--", lw=2, ms=8, color="#ea4335",
            label="полная задача трёх тел")
    ax.set_xscale("log")
    ax.set_xticks([20_000, 50_000, 100_000, 500_000])
    ax.set_xticklabels(["20 000", "50 000", "100 000", "500 000"])
    ax.minorticks_off()
    ax.set_xlabel("высота перицентра над Юлом, км")
    ax.set_ylabel("угол поворота δ (относительно Юла), °")
    ax.set_title("Модель сфер действия против полной задачи трёх тел")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(hs, [r["apo_pc"] for r in rows], "o-", lw=2, ms=8, color="#1a73e8",
            label="модель сфер действия")
    ax.plot(hs, [r["apo_3body"] for r in rows], "s--", lw=2, ms=8, color="#ea4335",
            label="полная задача трёх тел")
    ax.set_xscale("log")
    ax.set_xticks([20_000, 50_000, 100_000, 500_000])
    ax.set_xticklabels(["20 000", "50 000", "100 000", "500 000"])
    ax.minorticks_off()
    ax.set_xlabel("высота перицентра над Юлом, км")
    ax.set_ylabel("афелий после пролёта, млн км")
    ax.set_title("Насколько допущение искажает итоговую орбиту")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_conics_vs_3body.png"))
    plt.close(fig)
    print(f"\n  График: {os.path.join(OUT, 'fig7_conics_vs_3body.png')}")


if __name__ == "__main__":
    main()
