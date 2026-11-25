"""
Модель взлёта ракеты, ступенчатого отрыва ступеней и топливного баланса миссии.

Восхождение (Земля -> орбита 100 км):
  двухступенчатая ракета, вертикальный участок и программа крена
  gamma(h) = 90° * exp(-(h - h_tip) / H_PITCH); силы: тяга, гравитация,
  сопротивление воздуха (экспоненциальная атмосфера). Ступень сгорает ->
  отбрасывание сухой массы -> зажигание следующей. Бюджет Delta_v замыкается
  точно:  int(T/m dt) = v_final + int(g*sin(gamma) dt) + int(D/m dt).

Топливный баланс аппарата (по мере полёта):
  стек = аппарат (сухой) + манёвровое топливо (Оберт + резерв) + БУП
  (топливо и сухая масса для выхода на трансферную орбиту). Масса убывает
  только при импульсах (в связном полёте расхода нет): ТМI -> сброс БУП ->
  планирование (масса постоянна) -> импульс Оберта у Юпитера -> уход.

Числа ТМI и Оберта берутся из src/sim.py (единый источник истины).
Запуск:  python3 src/launch.py
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from constants import (
    OUT, DAY, G0, R_EARTH, MU_EARTH, RHO0, H_SCALE, CDA,
    STAGE1, STAGE2, H_TIP, H_PITCH, ASC_DT as DT, V_ORBIT_TARGET,
    M_DRY, ISP_MAN, DV_RESERVE, ISP_KICK, MS_KICK, T_KICK, T_PROBE,
    ATM_JUPITER, R_JUPITER,
)
from sim import launch_window, oberth_dv


def rho(h: float) -> float:
    """Экспоненциальная модель плотности воздуха."""
    return RHO0 * math.exp(-max(h, 0.0) / H_SCALE)


def grav(h: float) -> float:
    """Ускорение свободного падения на высоте h."""
    return G0 * (R_EARTH / (R_EARTH + max(h, 0.0))) ** 2


def gamma_cmd(h: float) -> float:
    """Программа крена: вертикальный участок, затем экспоненциальный крен к горизонту."""
    if h < H_TIP:
        return math.radians(90.0)
    return max(0.0, math.radians(90.0) * math.exp(-(h - H_TIP) / H_PITCH))


def v_circ(h: float) -> float:
    """Круговая орбитальная скорость на высоте h."""
    return math.sqrt(MU_EARTH / (R_EARTH + h))


def ascent(payload: float) -> dict:
    """Интегрирование ступенчатого восхождения с отбрасыванием ступеней.

    Возвращает события (сгорание, отрыв), траекторию и бюджет Delta_v.
    """
    h = v = x = t = 0.0
    dv_grav = dv_drag = 0.0
    m = STAGE1["m_prop"] + STAGE1["m_struct"] + STAGE2["m_prop"] + STAGE2["m_struct"] + payload
    m0 = m
    events = [dict(event="liftoff", t=0.0, h=0.0, v=0.0, gamma_deg=90.0, m=m)]
    track = [dict(t=0.0, h=0.0, v=0.0, gamma_deg=90.0, m=m)]
    stage_dv = []
    for st in (STAGE1, STAGE2):
        mp_left = st["m_prop"]
        mdot = st["T"] / (st["isp"] * G0)
        t_ign, m_ign = t, m
        while mp_left > 1e-6:
            gma = gamma_cmd(h)
            d = 0.5 * rho(h) * CDA * v * v
            v += (st["T"] / m - d / m - grav(h) * math.sin(gma)) * DT
            h += v * math.sin(gma) * DT
            x += v * math.cos(gma) * DT
            dv_grav += grav(h) * math.sin(gma) * DT
            dv_drag += (d / m) * DT
            dm = min(mdot * DT, mp_left)
            mp_left -= dm
            m -= dm
            t += DT
            if int(t / DT) % 25 == 0:
                track.append(dict(t=t, h=h, v=v, gamma_deg=math.degrees(gamma_cmd(h)), m=m))
        dv_stage = st["isp"] * G0 * math.log(m_ign / m)
        stage_dv.append(dv_stage)
        events.append(dict(event=f"cutoff {st['name']}", t=t, h=h, v=v,
                           gamma_deg=math.degrees(gamma_cmd(h)), m=m))
        track.append(dict(t=t, h=h, v=v, gamma_deg=math.degrees(gamma_cmd(h)), m=m))
        if st is STAGE1:
            m -= STAGE1["m_struct"]
            events.append(dict(event="отрыв ступени 1", t=t, h=h, v=v,
                               gamma_deg=math.degrees(gamma_cmd(h)), m=m))
            track.append(dict(t=t, h=h, v=v, gamma_deg=math.degrees(gamma_cmd(h)), m=m))
    return dict(
        m0=m0, events=events, track=track, stage_dv=stage_dv,
        dv_grav=dv_grav, dv_drag=dv_drag, dv_sum=sum(stage_dv),
        t_final=t, h_final=h, v_final=v,
        gamma_final_deg=math.degrees(gamma_cmd(h)), x_final=x, m_final=m,
        v_circ_final=v_circ(h),
    )


def spacecraft_stack(dv_tmi: float, dv_oberth: float) -> dict:
    """Топливный баланс аппарата: манёвровое топливо + блок выведения (БУП)."""
    ve_man = ISP_MAN * G0
    ve_kick = ISP_KICK * G0
    dv_man_total = dv_oberth + DV_RESERVE
    m_fuel_man = M_DRY * (math.exp(dv_man_total / ve_man) - 1.0)   # бак: сухой + манёвр
    m_probe_wet = M_DRY + m_fuel_man
    m_fuel_kick = (m_probe_wet + MS_KICK) * (math.exp(dv_tmi / ve_kick) - 1.0)
    m_stack = m_probe_wet + MS_KICK + m_fuel_kick
    burn_obert = (m_probe_wet) * (1.0 - math.exp(-dv_oberth / ve_man))
    return dict(
        m_dry=M_DRY, dv_man_total=dv_man_total, m_fuel_man=m_fuel_man,
        m_probe_wet=m_probe_wet, m_fuel_kick=m_fuel_kick, m_stack=m_stack,
        m_after_tmi=m_probe_wet + MS_KICK, m_after_sep=m_probe_wet,
        m_burn_obert=burn_obert, m_after_obert=m_probe_wet - burn_obert,
        t_tmi=m_fuel_kick * ve_kick / T_KICK,   # длительность импульса ТМI, с
        t_obert=m_probe_wet * dv_oberth / T_PROBE,   # длительность импульса Оберта, с
    )


def mission_mass_timeline(stack: dict, dv_tmi: float, dv_oberth: float, tof_days: float) -> dict:
    """Кусочно-линейная временная шкала массы аппарата по фазам полёта (с)."""
    t_orbit = 400.0                      # выход на орбиту (окончание 2-й ступени)
    t_hold = 1_800.0                     # блокировка на орбите до окна выхода
    t_tmi_end = t_hold + stack["t_tmi"]
    t_sep = t_tmi_end + 30.0             # сброс БУП после импульса ТМI
    t_flyby = tof_days * DAY
    t_obert_end = t_flyby + stack["t_obert"]
    t_end = t_flyby + 1_500.0 * DAY
    m0, m1 = stack["m_stack"], stack["m_after_tmi"]
    m2, m3 = stack["m_after_sep"], stack["m_probe_wet"]
    m4, m5 = stack["m_probe_wet"], stack["m_after_obert"]
    return dict(
        points=[
            dict(t=t_orbit, m=m0, label="выход на орбиту"),
            dict(t=t_hold, m=m0, label="блокировка"),
            dict(t=t_hold, m=m1, label=None),          # начало ТМI
            dict(t=t_tmi_end, m=m1, label=None),       # конец ТМI
            dict(t=t_sep, m=m2, label="сброс БУП"),
            dict(t=t_sep, m=m3, label=None),
            dict(t=t_flyby, m=m3, label="Юпитер"),
            dict(t=t_flyby, m=m4, label=None),         # начало Оберта
            dict(t=t_obert_end, m=m4, label=None),     # конец Оберта
            dict(t=t_obert_end, m=m5, label=None),
            dict(t=t_end, m=m5, label=None),
        ],
        t_tmi_start=t_hold, t_tmi_end=t_tmi_end, t_sep=t_sep,
        t_flyby=t_flyby, t_obert_end=t_obert_end, t_end=t_end,
    )


def fig_launch(asc: dict, path: str) -> None:
    """Рисунок 6: траектория восхождения и масса/скорость со событиями отрыва."""
    tr = np.array([[p["t"], p["h"] / 1e3, p["v"], p["m"]] for p in asc["track"]])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(tr[:, 1], tr[:, 2], color="tab:blue", lw=1.6, label="траектория")
    for ev in asc["events"]:
        if ev["event"] == "liftoff":
            ax1.plot([ev["h"] / 1e3], [ev["v"]], "o", color="tab:green", ms=7, zorder=5)
        elif "cutoff" in ev["event"]:
            ax1.plot([ev["h"] / 1e3], [ev["v"]], "s", color="tab:red", ms=7, zorder=5)
            ax1.annotate(ev["event"].replace("cutoff ", "сгорание "),
                         (ev["h"] / 1e3, ev["v"]), textcoords="offset points",
                         xytext=(8, -12), fontsize=9)
        elif "отрыв" in ev["event"]:
            ax1.plot([ev["h"] / 1e3], [ev["v"]], "^", color="tab:orange", ms=9, zorder=5)
            ax1.annotate("отрыв\nступени 1", (ev["h"] / 1e3, ev["v"]),
                         textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax1.axhline(v_circ(V_ORBIT_TARGET), ls="--", c="gray", lw=1,
                label="скорость орбиты 100 км")
    ax1.set_xlabel("высота h, км")
    ax1.set_ylabel("скорость v, м/с")
    ax1.set_title("Профиль восхождения")
    ax1.legend(fontsize=9, loc="lower right")

    t = tr[:, 0]
    ax2.plot(t, tr[:, 3] / 1e3, color="tab:purple", lw=1.6, label="масса, т")
    ax2.set_xlabel("время t, с")
    ax2.set_ylabel("масса, т", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax2b = ax2.twinx()
    ax2b.plot(t, tr[:, 2], color="tab:blue", lw=1.2, alpha=0.75, label="скорость, м/с")
    ax2b.set_ylabel("скорость, м/с", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    for ev in asc["events"]:
        if ev["event"] == "отрыв ступени 1":
            ax2.axvline(ev["t"], ls=":", c="tab:orange", lw=1.2)
            ax2.annotate("отрыв ступени 1", (ev["t"], ev["m"] / 1e3),
                         textcoords="offset points", xytext=(6, 14),
                         fontsize=9, color="tab:orange")
        if "cutoff" in ev["event"]:
            ax2.axvline(ev["t"], ls=":", c="tab:red", lw=1.0)
    ax2.set_title("Масса и скорость при взлёте")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_mass(tl: dict, stack: dict, tof_days: float, dv_tmi: float, path: str) -> None:
    """Рисунок 7: топливный баланс аппарата по фазам полёта (логарифмическая шкала)."""
    pts = tl["points"]
    ts = [p["t"] for p in pts]
    ms = [p["m"] for p in pts]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(ts, ms, drawstyle="steps-post", color="tab:blue", lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("время от выхода на орбиту, с (логарифмическая шкала)")
    ax.set_ylabel("масса аппарата, кг")
    ax.set_title("Топливный баланс аппарата по мере полёта")

    marks = [
        (tl["t_tmi_start"], stack["m_stack"], "ТМI", (14, -6), "left"),
        (tl["t_sep"], stack["m_probe_wet"], "сброс БУП", (14, 8), "left"),
        (tl["t_flyby"], stack["m_probe_wet"], "Оберт", (-8, 16), "right"),
        (tl["t_end"], stack["m_after_obert"], "уход", (-8, 16), "right"),
    ]
    for tt, mm, lab, off, ha in marks:
        ax.plot([tt], [mm], "o", color="tab:red", ms=6, zorder=5)
        ax.annotate(lab, (tt, mm), textcoords="offset points", xytext=off,
                    fontsize=9, ha=ha, color="tab:red")
    budget = (
        f"Фаза (масса аппарата):\n"
        f"орбита ожидания — {stack['m_stack']:.1f} кг\n"
        f"ТМI Δv {dv_tmi:.0f} м/с ({stack['t_tmi']:.0f} с): −{stack['m_fuel_kick']:.1f} кг "
        f"топлива БУП → {stack['m_after_tmi']:.1f} кг\n"
        f"сброс БУП: −180 кг → {stack['m_probe_wet']:.1f} кг\n"
        f"планирование {tof_days:.1f} сут: масса постоянна (расхода нет)\n"
        f"Оберт Δv 4.8 м/с: −{stack['m_burn_obert']:.2f} кг → {stack['m_after_obert']:.2f} кг "
        f"(резерв {stack['m_fuel_man'] - stack['m_burn_obert']:.1f} кг)"
    )
    ax.text(0.34, 0.72, budget, transform=ax.transAxes, fontsize=9,
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.95))
    ax.set_ylim(700.0, 13_000.0)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    tr = launch_window()
    _, dv_ob = oberth_dv(R_JUPITER + ATM_JUPITER, tr.v_inf_j, tr.v_jup)
    stack = spacecraft_stack(tr.dv_tmi, dv_ob)
    asc = ascent(stack["m_stack"])
    tl = mission_mass_timeline(stack, tr.dv_tmi, dv_ob, tr.tof / DAY)

    print("=" * 78)
    print("4. ВЗЛЁТ РАКЕТЫ И ОТБРАСЫВАНИЕ СТУПЕНЕЙ")
    print("=" * 78)
    print(f"  стек аппарата (полезная нагрузка): {stack['m_stack']:.1f} кг")
    print(f"    сухой аппарат {stack['m_dry']:.0f} + манёвровое топливо {stack['m_fuel_man']:.1f} "
          f"+ БУП {MS_KICK:.0f} + топливо БУП {stack['m_fuel_kick']:.1f}")
    print(f"  ракета: С1 (тяга {STAGE1['T']/1e6:.0f} МН, Isp {STAGE1['isp']:.0f} с, "
          f"топливо {STAGE1['m_prop']:,.0f} кг) + С2 (тяга {STAGE2['T']/1e6:.1f} МН, "
          f"Isp {STAGE2['isp']:.0f} с, топливо {STAGE2['m_prop']:,.0f} кг)")
    print(f"  стартовая масса: {asc['m0']:,.0f} кг, тяга/вес = "
          f"{STAGE1['T'] / (asc['m0'] * G0):.2f}")
    for ev in asc["events"][1:]:
        print(f"    {ev['event']:<22} t = {ev['t']:6.1f} с, h = {ev['h']/1e3:6.1f} км, "
              f"v = {ev['v']:6.0f} м/с, γ = {ev['gamma_deg']:5.1f}°, m = {ev['m']:8,.0f} кг")
    print(f"  выход на орбиту: h = {asc['h_final']/1e3:.1f} км, v = {asc['v_final']:.0f} м/с "
          f"(круговая {asc['v_circ_final']:.0f} м/с, Δ = {asc['v_final'] - asc['v_circ_final']:+.1f}), "
          f"γ = {asc['gamma_final_deg']:.1f}°")
    print(f"  бюджет Δv: С1 {asc['stage_dv'][0]:.0f} + С2 {asc['stage_dv'][1]:.0f} "
          f"= {asc['dv_sum']:.0f} м/с")
    print(f"    = v_итог {asc['v_final']:.0f} + потери на гравитацию {asc['dv_grav']:.0f} "
          f"+ потери на сопротивление {asc['dv_drag']:.0f} "
          f"(замыкание {asc['v_final'] + asc['dv_grav'] + asc['dv_drag'] - asc['dv_sum']:+.1f} м/с)")

    print("\n" + "=" * 78)
    print("5. ТОПЛИВНЫЙ БАЛАНС АППАРАТА ПО МЕРЕ ПОЛЁТА")
    print("=" * 78)
    print(f"  орбита ожидания:      {stack['m_stack']:.1f} кг (сухой + манёвр + БУП)")
    print(f"  импульс ТМI (Δv {tr.dv_tmi:.1f} м/с, {stack['t_tmi']:.0f} с): "
          f"−{stack['m_fuel_kick']:.1f} кг топлива БУП → {stack['m_after_tmi']:.1f} кг")
    print(f"  сброс БУП:            −{MS_KICK:.0f} кг → {stack['m_probe_wet']:.1f} кг")
    print(f"  планирование {tr.tof/DAY:.1f} сут: масса постоянна (в связном полёте топливо не расходуется)")
    print(f"  Оберт у Юпитера (Δv {dv_ob:.2f} м/с): −{stack['m_burn_obert']:.2f} кг манёврового "
          f"топлива → {stack['m_after_obert']:.2f} кг")
    print(f"  резерв на коррекции: {stack['m_fuel_man'] - stack['m_burn_obert']:.2f} кг "
          f"(Δv до {DV_RESERVE:.0f} м/с)")

    fig_launch(asc, os.path.join(OUT, "fig6_launch.png"))
    fig_mass(tl, stack, tr.tof / DAY, tr.dv_tmi, os.path.join(OUT, "fig7_mass.png"))
    out = dict(
        rocket=dict(stage1=STAGE1, stage2=STAGE2, h_tip=H_TIP, h_pitch=H_PITCH,
                    cd_a=CDA, gamma_program="90°·exp(−(h−h_tip)/H_PITCH)"),
        ascent={k: v for k, v in asc.items() if k != "track"},
        spacecraft=stack,
        mission=dict(tmi_dv_m_s=tr.dv_tmi, oberth_dv_m_s=dv_ob,
                     tof_days=tr.tof / DAY, timeline=tl),
    )
    with open(os.path.join(OUT, "launch.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nМодель взлёта и баланса сохранены в: {OUT} (fig6_launch.png, fig7_mass.png, launch.json)")


if __name__ == "__main__":
    main()
