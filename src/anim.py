"""
Анимация миссии Земля → Юпитер в Солнечной системе (out/mission.mp4).

Сценарий (16 с, 30 кадров/с) — модельный расчёт на реальных константах JPL:
  1) Фаза 1 (0–5.5 с): гомановский перелёт Земля → Юпитер (~998 сут).
     Земля и Юпитер вращаются по круговым орбитам с реальным временем миссии t
     (Юпитер из угла опережения 97.2° приходит в точку встречи с КА).
  2) Фаза 2 (5.5–10.5 с): пролёт Юпитера у границы атмосферы
     (h_p = 200 км, поворот δ = 158.41°) + импульс Оберта Δv ≈ 4.8 м/с в перицентре
     (самосогласованный расчёт: после импульса δ' ≈ 158.2°).
  3) Фаза 3 (10.5–16 с): уход из Солнечной системы!
     Показаны обе ветви после манёвра:
       - пунктиром: пассивный заброс к афелию Сатурна (h_p ≈ 1.37 млн км, Δv = 0);
       - основная траектория КА: разомкнутая траектория ухода за пределы всех орбит
         планет и притяжения Солнца (|v_вых| ≥ v_esc = 18 465 м/с).

Запуск:  python3 src/anim.py [--out out/mission.mp4] [--fps 30] [--seconds 16]
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib

try:
    import imageio_ffmpeg

    os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ["PATH"]
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from sim import (MU_SUN, MU_JUPITER, R_JUPITER, A_JUPITER, A_EARTH, A_SATURN, E_SATURN, ATM_JUPITER,
                 launch_window, delta_hyp, turn_vinf, oberth_dv, helio_elements, post_flyby,
                 rp_for_apo, OUT)

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

DAY = 86_400.0
OMEGA_EARTH = math.sqrt(MU_SUN / A_EARTH ** 3)
OMEGA_JUPITER = math.sqrt(MU_SUN / A_JUPITER ** 3)


def kepler_ellipse_pos(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float = MU_SUN) -> np.ndarray:
    """Точное аналитическое положение на эллипсе (без универсальных переменных)."""
    r_norm = float(np.linalg.norm(r0))
    v_norm = float(np.linalg.norm(v0))
    eps = 0.5 * v_norm ** 2 - mu / r_norm
    a = -mu / (2.0 * eps)
    e_vec = ((v_norm ** 2 - mu / r_norm) * r0 - float(np.dot(r0, v0)) * v0) / mu
    e = float(np.linalg.norm(e_vec))
    omega = math.atan2(float(e_vec[1]), float(e_vec[0]))
    theta0 = math.atan2(float(r0[1]), float(r0[0])) - omega
    E0 = 2.0 * math.atan2(math.sqrt(1.0 - e) * math.sin(0.5 * theta0),
                          math.sqrt(1.0 + e) * math.cos(0.5 * theta0))
    M0 = E0 - e * math.sin(E0)
    M = (M0 + math.sqrt(mu / a ** 3) * dt) % (2.0 * math.pi)
    E = M if e < 0.8 else math.pi
    for _ in range(45):
        E -= (E - e * math.sin(E) - M) / (1.0 - e * math.cos(E))
    x_orb = a * (math.cos(E) - e)
    y_orb = a * math.sqrt(1.0 - e ** 2) * math.sin(E)
    c, s = math.cos(omega), math.sin(omega)
    return np.array([c * x_orb - s * y_orb, s * x_orb + c * y_orb])


def kepler_hyperbola_pos(r0: np.ndarray, v0: np.ndarray, dt: float, mu: float = MU_SUN) -> np.ndarray:
    """Точное аналитическое положение на гелиоцентрической гиперболе ухода из системы (e > 1)."""
    r_norm = float(np.linalg.norm(r0))
    v_norm = float(np.linalg.norm(v0))
    eps = 0.5 * v_norm ** 2 - mu / r_norm
    a_abs = mu / (2.0 * eps)
    e_vec = ((v_norm ** 2 - mu / r_norm) * r0 - float(np.dot(r0, v0)) * v0) / mu
    e = float(np.linalg.norm(e_vec))
    h_z = float(r0[0] * v0[1] - r0[1] * v0[0])
    sign_h = 1.0 if h_z >= 0 else -1.0
    omega = math.atan2(float(e_vec[1]), float(e_vec[0]))

    # Проекции r0 в орбитальной системе (ось +x вдоль вектора эксцентриситета к перицентру)
    c, s = math.cos(omega), math.sin(omega)
    x0_orb = c * float(r0[0]) + s * float(r0[1])
    y0_orb = (-s * float(r0[0]) + c * float(r0[1])) * sign_h
    sinh_H0 = y0_orb / (a_abs * math.sqrt(e ** 2 - 1.0))
    H0 = math.asinh(sinh_H0)
    M0 = e * math.sinh(H0) - H0

    M = M0 + math.sqrt(mu / a_abs ** 3) * dt
    H = math.asinh(M / e)
    for _ in range(55):
        H -= (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1.0)
    x_orb = a_abs * (e - math.cosh(H))
    y_orb = sign_h * a_abs * math.sqrt(e ** 2 - 1.0) * math.sinh(H)
    return np.array([c * x_orb - s * y_orb, s * x_orb + c * y_orb])


def hyperbola_local(rp: float, e: float, t: float, mu: float = MU_JUPITER) -> np.ndarray:
    """
    Положение КА относительно Юпитера (2D) в момент времени t от перицентра.
    В точке встречи (-A_JUPITER, 0) Юпитер летит вниз (v_J = (0, -v_j)), а КА входит снизу вдоль +y,
    огибает Юпитер сзади по орбите (сверху) на высоте h_p = 200 км и разворачивается на 158.41°!
    """
    a_abs = rp / (e - 1.0)
    n = math.sqrt(mu / a_abs ** 3)
    M = n * t
    H = math.asinh(M / max(e, 1.0001))
    for _ in range(50):
        H -= (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1.0)
    psi_asym = math.acos(-1.0 / e)
    rot_ang = 0.5 * math.pi - (math.pi - psi_asym)
    x0 = a_abs * (e - math.cosh(H))
    y0 = a_abs * math.sqrt(e ** 2 - 1.0) * math.sinh(H)
    c, s = math.cos(rot_ang), math.sin(rot_ang)
    return np.array([c * x0 - s * y0, s * x0 + c * y0])


def planet_positions(t_mission: float, tr_tof: float):
    """Положения Земли и Юпитера в физический момент времени t_mission."""
    ang_k = OMEGA_EARTH * t_mission
    ang_j = math.pi + OMEGA_JUPITER * (t_mission - tr_tof)
    pos_k = np.array([A_EARTH * math.cos(ang_k), A_EARTH * math.sin(ang_k)])
    pos_j = np.array([A_JUPITER * math.cos(ang_j), A_JUPITER * math.sin(ang_j)])
    return pos_k, pos_j, ang_j


def build_scene(n_tr: int = 900, n_fly: int = 1200, n_post: int = 1200):
    """Строит полную миссию: перелёт -> пролёт Юпитера (h_p = 200 км + Оберт) -> выход из системы."""
    tr = launch_window()
    h_p_km = ATM_JUPITER / 1e3  # 200 км (граница атмосферы Юпитера)
    rp = R_JUPITER + ATM_JUPITER

    # Фаза 1: перелёт Земля -> Юпитер
    t_tr = np.linspace(0.0, tr.tof, n_tr)
    r0_tr = np.array([A_EARTH, 0.0])
    v0_tr = np.array([0.0, math.sqrt(MU_SUN * (2.0 / A_EARTH - 1.0 / tr.a_t))])
    path_tr = np.array([kepler_ellipse_pos(r0_tr, v0_tr, t) for t in t_tr])

    # Фаза 2: пролёт вблизи Юпитера (окно ±1.8 сут вокруг перицентра).
    # Импульс Оберта в перицентре: входящая ветвь (t < 0) — пассивная гипербола,
    # исходящая (t >= 0) — гипербола с увеличенной v_inf. Перицентр при этом
    # неподвижен: исходящая ветвь поворачивается так, чтобы точка перицентра
    # совпала с входящей (импульс направлен вдоль скорости).
    vinf_esc, dv_oberth = oberth_dv(rp, tr.v_inf_j, tr.v_jup)
    # Небольшой запас (+2%), чтобы гелиоцентрическая орбита была явно гиперболической.
    vinf_boosted = vinf_esc * 1.02
    e_h = 1.0 + rp * tr.v_inf_j ** 2 / MU_JUPITER
    e_out = 1.0 + rp * vinf_boosted ** 2 / MU_JUPITER
    rot_in = math.acos(-1.0 / e_h) - 0.5 * math.pi
    rot_out = math.acos(-1.0 / e_out) - 0.5 * math.pi
    d_rot = rot_in - rot_out
    c_rot, s_rot = math.cos(d_rot), math.sin(d_rot)
    t_half = 1.8 * DAY
    dt_fly = np.linspace(-t_half, t_half, n_fly)
    t_fly = tr.tof + dt_fly
    loc_list = []
    for dt in dt_fly:
        if dt < 0.0:
            loc_list.append(hyperbola_local(rp, e_h, dt))
        else:
            loc = hyperbola_local(rp, e_out, dt)
            loc_list.append(np.array([c_rot * loc[0] - s_rot * loc[1],
                                      s_rot * loc[0] + c_rot * loc[1]]))
    loc_fly = np.array(loc_list)
    path_fly = []
    for dt, loc in zip(dt_fly, loc_fly):
        _, jpos, ang_j = planet_positions(tr.tof + dt, tr.tof)
        d_ang = ang_j - math.pi
        c, s = math.cos(d_ang), math.sin(d_ang)
        path_fly.append(jpos + np.array([c * loc[0] - s * loc[1], s * loc[0] + c * loc[1]]))
    path_fly = np.array(path_fly)

    # Фаза 3: УХОД ИЗ СИСТЕМЫ ЗВЕЗДЫ (пролёт h_p = 200 км + импульс Оберта в перицентре).
    # Гелиоцентрическая ветвь стартует непрерывно от конца фазы 2 (без телепорта
    # обратно в точку встречи): начальное состояние — фактическое положение КА
    # в конце пролёта + скорость Юпитера в этот момент + повёрнутая v_inf'.
    d_max = delta_hyp(rp, tr.v_inf_j)
    d_boosted = delta_hyp(rp, vinf_boosted)
    vinf_out_std = turn_vinf(np.array([0.0, -vinf_boosted]), d_boosted, +1)
    v_out_meet = -(np.array([0.0, tr.v_jup]) + vinf_out_std)
    r_meet = np.array([-A_JUPITER, 0.0])
    v_esc_sun = math.sqrt(2.0 * MU_SUN / A_JUPITER)
    r_start = path_fly[-1].copy()
    _, _, ang_end = planet_positions(tr.tof + t_half, tr.tof)
    d_ang_end = ang_end - math.pi
    c_end, s_end = math.cos(d_ang_end), math.sin(d_ang_end)
    v_start = np.array([c_end * v_out_meet[0] - s_end * v_out_meet[1],
                        s_end * v_out_meet[0] + c_end * v_out_meet[1]])
    eps_sun = 0.5 * float(v_start @ v_start) - MU_SUN / float(np.linalg.norm(r_start))

    # Продолжаем полёт на 1500 суток — КА пробивает орбиту Сатурна (1433.5 млн км)
    # и уходит за пределы системы (в реальном полёте «Вояджера-2» это заняло годы).
    dt_post = np.linspace(0.0, 1500.0 * DAY, n_post)
    t_post = tr.tof + t_half + dt_post
    path_post = np.array([kepler_hyperbola_pos(r_start, v_start, dt) for dt in dt_post])

    # Дополнительно для сравнения на фоне: пассивная замкнутая орбита
    # к афелию Сатурна (h_p подобран так, чтобы афелий = 1514.5 млн км, Δv = 0)
    rp_saturn = rp_for_apo(A_SATURN * (1.0 + E_SATURN), tr.v_inf_j, +1, tr.v_jup)
    _, v_saturn_std = post_flyby(rp_saturn, tr.v_inf_j, +1, tr.v_jup)
    v_saturn_meet = -v_saturn_std
    el_saturn = helio_elements(r_meet, v_saturn_meet)
    T_saturn = 2.0 * math.pi * math.sqrt(el_saturn["a"] ** 3 / MU_SUN)
    path_saturn_ref = np.array([kepler_ellipse_pos(r_meet, v_saturn_meet, dt)
                               for dt in np.linspace(0.0, T_saturn, 600)])

    return dict(
        tr=tr, rp=rp, h_p_km=h_p_km, e_h=e_h, e_out=e_out,
        t_tr=t_tr, path_tr=path_tr,
        dt_fly=dt_fly, t_fly=t_fly, loc_fly=loc_fly, path_fly=path_fly,
        dt_post=dt_post, t_post=t_post, path_post=path_post,
        path_saturn_ref=path_saturn_ref,
        v_out_norm=float(np.linalg.norm(v_start)),
        eps_sun=eps_sun,
        v_esc_sun=v_esc_sun,
        dv_oberth=dv_oberth,
        delta_deg=math.degrees(d_max),
        delta_boosted_deg=math.degrees(d_boosted),
    )


def make(out_path: str, fps: int = 30, seconds: int = 16):
    sc = build_scene()
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.055, 0.13, 0.60, 0.80])
    ax.set_aspect("equal")
    # Расширенный масштаб: орбита Сатурна (1433.5 млн км) и уход КА за пределы системы
    ax.set_xlim(-2350, 2350)
    ax.set_ylim(-1980, 1420)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Гравитационный манёвр у Юпитера и уход КА за пределы планетной системы")

    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(A_EARTH * np.cos(th) / 1e9, A_EARTH * np.sin(th) / 1e9,
            color="#8ab4f8", lw=1.1, label="орбита Земли")
    ax.plot(A_JUPITER * np.cos(th) / 1e9, A_JUPITER * np.sin(th) / 1e9,
            color="#34a853", lw=1.2, label="орбита Юпитера")
    # Внешняя орбита модели — Сатурн
    b_ee = A_SATURN * math.sqrt(1.0 - E_SATURN ** 2)
    xe, ye = A_SATURN * np.cos(th) - A_SATURN * E_SATURN, b_ee * np.sin(th)
    om = math.radians(260)
    ax.plot((xe * math.cos(om) - ye * math.sin(om)) / 1e9,
            (xe * math.sin(om) + ye * math.cos(om)) / 1e9,
            color="#9aa0a6", lw=1.1, ls="--", label="орбита Сатурна")
    # Справочная замкнутая ветвь без импульса Оберта
    ax.plot(sc["path_saturn_ref"][:, 0] / 1e9, sc["path_saturn_ref"][:, 1] / 1e9,
            color="#9334e6", lw=1.0, ls=":", alpha=0.65, label="пассивный заброс к Сатурну (Δv=0)")
    ax.plot([0], [0], "o", ms=13, color="#fbbc04", label="Солнце (звезда)")

    jupiter_trail, = ax.plot([], [], color="#34a853", lw=2.5, alpha=0.35, label="путь Юпитера по орбите")
    line_t, = ax.plot([], [], color="#ea4335", lw=1.9, label="перелёт Земля → Юпитер")
    line_p, = ax.plot([], [], color="#e37400", lw=2.3, label="траектория ухода из системы!")

    earth_dot, = ax.plot([], [], "o", ms=6.5, color="#1a73e8", zorder=4, label="Земля")
    jupiter_dot, = ax.plot([], [], "o", ms=10.5, color="#34a853", mec="#1b5e20", mew=1.1, zorder=5, label="Юпитер")
    ship, = ax.plot([], [], "o", ms=8.5, color="white", mec="#202124", mew=1.5, zorder=6, label="КА")
    ax.legend(loc="upper right", fontsize=7.3)

    # Врезка: крупный план гиперболы у Юпитера (h_p = 200 км, поворот 158.41°)
    axz = fig.add_axes([0.70, 0.56, 0.28, 0.36])
    axz.set_aspect("equal", adjustable="box")
    axz.autoscale(False)
    zoom_kkm = 700.0
    axz.set_xlim(-zoom_kkm, zoom_kkm)
    axz.set_ylim(-zoom_kkm, zoom_kkm)
    axz.set_title("Система Юпитера (h_p = 200 км, δ = 158°)", fontsize=9.2)
    axz.tick_params(labelsize=7.5)
    axz.set_xlabel("x, тыс. км", fontsize=8)
    axz.set_ylabel("y, тыс. км", fontsize=8)
    axz.add_patch(plt.Circle((0, 0), R_JUPITER / 1e6, color="#34a853", alpha=0.9, zorder=3))
    axz.annotate("", xy=(0, -330), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.6))
    axz.text(28, -320, "v_Юпитера", fontsize=8, color="#1b5e20")
    axz.plot(sc["loc_fly"][:, 0] / 1e6, sc["loc_fly"][:, 1] / 1e6,
             color="#e37400", lw=1.0, alpha=0.30, ls="--", zorder=1)
    zline, = axz.plot([], [], color="#e37400", lw=2.0, zorder=4)
    zship, = axz.plot([], [], "o", ms=8.5, color="#ea4335", mec="white", mew=1.2, zorder=6)
    burn_flash, = axz.plot([], [], "*", ms=15, color="#fbbc04", mec="#e37400", zorder=7)

    txt = fig.text(0.70, 0.50, "", va="top", fontsize=9.8, family="monospace")
    phase_txt = fig.text(0.70, 0.045, "", fontsize=9.8, weight="bold")
    bar = fig.add_axes([0.055, 0.055, 0.60, 0.015])
    bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, color="#e8eaed"))
    prog = bar.add_patch(plt.Rectangle((0, 0), 0, 1, color="#1a73e8"))

    T1, T2, T3 = 5.5, 5.0, 5.5
    total = T1 + T2 + T3
    jupiter_history = []

    def frame(i: int):
        n_frames = fps * seconds
        t_scr = total * i / max(1, n_frames - 1)

        if t_scr <= T1:
            u = t_scr / T1
            k = int(u * (len(sc["t_tr"]) - 1))
            t_mis = float(sc["t_tr"][k])
            pos = sc["path_tr"][k]
            line_t.set_data(sc["path_tr"][:k + 1, 0] / 1e9, sc["path_tr"][:k + 1, 1] / 1e9)
            pos_k, pos_j, ang_j = planet_positions(t_mis, sc["tr"].tof)
            burn_flash.set_data([], [])
            phase = "Фаза 1: гомановский перелёт Земля → Юпитер\n(обе планеты движутся к точке встречи)"
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"долгота Юпитера : {math.degrees(ang_j)%360:7.1f}°\n"
                f"расст. до КА  : {np.linalg.norm(pos)/1e9:7.2f} млн км\n"
                f"Δv старта     : {sc['tr'].dv_tmi:7.0f} м/с\n"
                f"v прилёта     : {sc['tr'].v_arr:7.0f} м/с"
            )
        elif t_scr <= T1 + T2:
            u = (t_scr - T1) / T2
            k = int(u * (len(sc["t_fly"]) - 1))
            t_mis = float(sc["t_fly"][k])
            pos = sc["path_fly"][k]
            loc = sc["loc_fly"][k] / 1e6
            zline.set_data(sc["loc_fly"][:k + 1, 0] / 1e6, sc["loc_fly"][:k + 1, 1] / 1e6)
            zship.set_data([loc[0]], [loc[1]])
            if abs(sc["dt_fly"][k]) < 0.14 * DAY:
                burn_flash.set_data([loc[0]], [loc[1]])
            else:
                burn_flash.set_data([], [])
            pos_k, pos_j, ang_j = planet_positions(t_mis, sc["tr"].tof)
            h_now = (np.linalg.norm(sc["loc_fly"][k]) - R_JUPITER) / 1e3
            phase = (f"Фаза 2: праща Юпитера (h_p = 200 км, δ = {sc['delta_deg']:.0f}°)\n"
                     f"+ импульс Оберта Δv = {sc['dv_oberth']:.1f} м/с в перицентре!")
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"высота над Юпит.: {h_now:7.0f} км\n"
                f"перицентр h_p : {sc['h_p_km']:7.0f} км\n"
                f"поворот δ     : {sc['delta_deg']:7.2f}°\n"
                f"импульс Оберта: {sc['dv_oberth']:7.1f} м/с"
            )
        else:
            u = (t_scr - T1 - T2) / T3
            k = int(u * (len(sc["t_post"]) - 1))
            t_mis = float(sc["t_post"][k])
            pos = sc["path_post"][k]
            line_p.set_data(sc["path_post"][:k + 1, 0] / 1e9, sc["path_post"][:k + 1, 1] / 1e9)
            burn_flash.set_data([], [])
            pos_k, pos_j, ang_j = planet_positions(t_mis, sc["tr"].tof)
            r_now = float(np.linalg.norm(pos))
            v_loc_esc = math.sqrt(2.0 * MU_SUN / r_now)
            # Текущая скорость КА из закона сохранения гелиоцентрической энергии
            v_now = math.sqrt(2.0 * (sc["eps_sun"] + MU_SUN / r_now))
            phase = "Фаза 3: УХОД ИЗ СОЛНЕЧНОЙ СИСТЕМЫ!\n(КА пересекает все орбиты с v > v_esc)"
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"расст. от зв. : {r_now/1e9:7.1f} млн км\n"
                f"скорость КА   : {v_now:7.0f} м/с\n"
                f"2-я косм. зв. : {v_loc_esc:7.0f} м/с\n"
                f"статус: УХОД ИЗ СИСТЕМЫ (v > v_esc)"
            )

        jupiter_history.append(pos_j / 1e9)
        jh = np.array(jupiter_history)
        jupiter_trail.set_data(jh[:, 0], jh[:, 1])

        earth_dot.set_data([pos_k[0] / 1e9], [pos_k[1] / 1e9])
        jupiter_dot.set_data([pos_j[0] / 1e9], [pos_j[1] / 1e9])
        ship.set_data([pos[0] / 1e9], [pos[1] / 1e9])
        prog.set_width(i / max(1, n_frames - 1))
        phase_txt.set_text(phase)
        txt.set_text(info)
        return []

    anim = FuncAnimation(fig, frame, frames=fps * seconds, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2600,
                          metadata={"title": "Jupiter Gravity Assist & System Escape"})
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Видео сохранено: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "mission.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=16)
    args = ap.parse_args()
    make(args.out, fps=args.fps, seconds=args.seconds)
