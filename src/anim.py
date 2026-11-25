"""
Анимация миссии Земля -> Юпитер в Солнечной системе (out/mission.mp4):
проекция на плоскость эклиптики (орбита Земли); наклонение орбиты Юпитера
1.303° в анимации не показывается — его влияние на модель учтено в src/sim.py
(функция coplanarity, поправки <=0.03 %).

Непрерывный сценарий полёта — от старта с Земли до выхода за пределы
Солнечной системы (22 с, 30 кадров/с), модельный расчёт на константах
src/constants.py:
  0) Фаза 0 (0–4 с): старт с Земли. В правой врезке — «Система Земли»:
     ракета идёт от поверхности до круговой орбиты 100 км по программе
     крена, отбрасывается ступень 1 (высоты преувеличены ×8 для
     наглядности); левый план — положение КА у Земли.
  1) Фаза 1 (4–10 с): межпланетный перелёт: КА летит к рассчитанной
     точке встречи с Юпитером; справа плавно появляется кадр
     «Система Юпитера» (рассчитанная гипербола пролёта).
  2) Фаза 2 (10–17 с): пролёт Юпитера. Переход плавный: левый план
     наезжает на Юпитер, время у перицентра замедляется (sin-маппинг),
     врезка приближается (±700 -> ±280 тыс. км), чтобы КА не
     «касался» Юпитера (h_p = 200 км, δ = 158°, импульс Оберта).
  3) Фаза 3 (17–22 с): уход из Солнечной системы (v > v_esc).

Запуск:  python3 src/anim.py [--out out/mission.mp4] [--fps 30] [--seconds 22]
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
from matplotlib.patches import Circle

from constants import (MU_SUN, MU_JUPITER, R_JUPITER, A_JUPITER, A_EARTH, A_SATURN,
                       E_SATURN, ATM_JUPITER, R_EARTH, V_ORBIT_TARGET, DAY,
                       OMEGA_EARTH, OMEGA_JUPITER, OUT)
from sim import launch_window, delta_hyp, turn_vinf, oberth_dv
from launch import ascent, spacecraft_stack

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

# Длительности фаз, с (сумма =22 с видео по умолчанию)
T0, T1, T2, T3 = 4.0, 6.0, 7.0, 5.0
TOTAL = T0 + T1 + T2 + T3
# Преувеличение высоты в земной врезке (орбита 100 км визуально отделена от поверхности)
EXAG = 20.0
# Соотношение сторон левого плана (ширина/высота его axes) — держим постоянным,
# чтобы при наезде камеры не «прыгал» прямоугольник графика
MAIN_ASPECT = 4700.0 / 3400.0


def sstep(x: float) -> float:
    """гладкий шаг0..1 (3x² − 2x³)."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


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
    """Строит непрерывную миссию: вывод на орбиту Земли -> перелёт -> пролёт Юпитера -> уход."""
    tr = launch_window()
    h_p_km = ATM_JUPITER / 1e3  # 200 км (граница атмосферы Юпитера)
    rp = R_JUPITER + ATM_JUPITER

    # Фаза 0: восхождение ракеты (Земля -> орбита 100 км), данные launch.ascent
    vinf_esc, dv_oberth = oberth_dv(rp, tr.v_inf_j, tr.v_jup)
    stack = spacecraft_stack(tr.dv_tmi, dv_oberth)
    asc = ascent(stack["m_stack"])
    track = asc["track"]
    # Интегрирование угла вниз по треку: dphi = v*cos(gamma)*dt / r (истинный радиус)
    phi = math.pi / 2.0          # старт с «верхней» точки диска Земли
    asc_pts = []
    first = track[0]
    r_draw = (R_EARTH + EXAG * first["h"]) / 1e6   # тыс. км
    asc_pts.append(dict(x=r_draw * math.cos(phi), y=r_draw * math.sin(phi),
                        t=first["t"], h=first["h"], v=first["v"], m=first["m"],
                        gamma_deg=first["gamma_deg"]))
    prev = first
    for row in track[1:]:
        ddt = row["t"] - prev["t"]
        v_bar = 0.5 * (row["v"] + prev["v"])
        g_bar = math.radians(0.5 * (row["gamma_deg"] + prev["gamma_deg"]))
        r_bar = 0.5 * (2.0 * R_EARTH + row["h"] + prev["h"])
        if r_bar > 0.0 and ddt > 0.0:
            phi += v_bar * math.cos(g_bar) * ddt / r_bar
        r_draw = (R_EARTH + EXAG * row["h"]) / 1e6   # тыс. км
        asc_pts.append(dict(x=r_draw * math.cos(phi), y=r_draw * math.sin(phi),
                            t=row["t"], h=row["h"], v=row["v"], m=row["m"],
                            gamma_deg=row["gamma_deg"]))
        prev = row
    asc_path = np.array([[p["x"], p["y"]] for p in asc_pts])
    # Индекс отрыва ступени 1 (последнее падение массы в треке)
    sep_idx = 0
    for i in range(1, len(asc_pts)):
        if asc_pts[i - 1]["m"] - asc_pts[i]["m"] > 1_000.0:   # отброс сухой массы ступени
            sep_idx = i
            break
    # Выходная орбита: дуга после выведения (та же скорость отсчёта, что у восхождения)
    r_c = (R_EARTH + EXAG * asc["h_final"]) / 1e6
    v_c = asc["v_final"]
    rate = asc["t_final"] / (0.8 * T0)              # модельных с на секунду экрана
    coast_dphi = (v_c / (R_EARTH + asc["h_final"])) * (0.2 * T0) * rate
    n_coast = 40
    phi_end = phi
    coast_path = np.array([[r_c * math.cos(phi_end + coast_dphi * i / (n_coast - 1)),
                            r_c * math.sin(phi_end + coast_dphi * i / (n_coast - 1))]
                           for i in range(n_coast)])

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
    e_h = 1.0 + rp * tr.v_inf_j ** 2 / MU_JUPITER
    e_out = 1.0 + rp * vinf_esc ** 2 / MU_JUPITER
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

    # Фаза 3: уход из системы звезды (пролёт h_p = 200 км + импульс Оберта в перицентре).
    # Гелиоцентрическая ветвь стартует непрерывно от конца фазы 2 (без телепорта
    # обратно в точку встречи): начальное состояние — фактическое положение КА
    # в конце пролёта + скорость Юпитера в этот момент + повёрнутая v_inf'.
    d_max = delta_hyp(rp, tr.v_inf_j)
    d_boosted = delta_hyp(rp, vinf_esc)
    vinf_out_std = turn_vinf(np.array([0.0, -vinf_esc]), d_boosted, +1)
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

    # Продолжаем полёт на1500 суток — КА пробивает орбиту Сатурна (1433.5 млн км)
    # и уходит за пределы системы (в реальном полёте «Вояджера-2» это заняло годы).
    dt_post = np.linspace(0.0, 1500.0 * DAY, n_post)
    t_post = tr.tof + t_half + dt_post
    path_post = np.array([kepler_hyperbola_pos(r_start, v_start, dt) for dt in dt_post])

    return dict(
        tr=tr, rp=rp, h_p_km=h_p_km, e_h=e_h, e_out=e_out,
        asc=asc, asc_pts=asc_pts, asc_path=asc_path, coast_path=coast_path,
        sep_idx=sep_idx, stack=stack,
        t_tr=t_tr, path_tr=path_tr,
        t_half=t_half, dt_fly=dt_fly, t_fly=t_fly, loc_fly=loc_fly, path_fly=path_fly,
        dt_post=dt_post, t_post=t_post, path_post=path_post,
        r_meet=r_meet,
        v_out_norm=float(np.linalg.norm(v_start)),
        eps_sun=eps_sun,
        v_esc_sun=v_esc_sun,
        dv_oberth=dv_oberth,
        delta_deg=math.degrees(d_max),
        delta_boosted_deg=math.degrees(d_boosted),
    )


def _fade_artists(ax):
    """Все художники врезки, включая метки-тики и стрелки annotate."""
    arts = list(ax.lines) + list(ax.patches) + list(ax.texts)
    for t in ax.texts:
        ap = getattr(t, "arrow_patch", None)
        if ap is not None:
            arts.append(ap)
    for axis in (ax.xaxis, ax.yaxis):
        arts.extend(axis.get_ticklines())
    return arts


def _alpha_base(ax) -> dict:
    """Базовые прозрачности художников врезки (для плавных переходов)."""
    return {id(c): (c.get_alpha() if c.get_alpha() is not None else 1.0)
            for c in _fade_artists(ax)}


def set_ax_alpha(ax, base: dict, a: float) -> None:
    """Плавно показать/скрыть врезку целиком (фон, рамка, подписи, содержимое)."""
    ax.patch.set_alpha(a)
    for sp in ax.spines.values():
        sp.set_alpha(a)
    ax.title.set_alpha(a)
    ax.xaxis.label.set_alpha(a)
    ax.yaxis.label.set_alpha(a)
    for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        t.set_alpha(a)
    for c in _fade_artists(ax):
        c.set_alpha(a * base.get(id(c), 1.0))


def make(out_path: str, fps: int = 30, seconds: int = 22):
    sc = build_scene()
    tr = sc["tr"]
    asc = sc["asc"]
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.055, 0.13, 0.60, 0.80])
    ax.set_aspect("equal")
    ax.set_xlim(-2350, 2350)
    ax.set_ylim(-1980, 1420)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Непрерывный полёт: старт с Земли -> орбита -> Юпитер -> уход из системы")

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
    # Рассчитанная точка встречи с Юпитером
    ax.plot([-A_JUPITER / 1e9], [0.0], marker="X", ms=9, color="#ea4335",
            mec="#202124", mew=0.8, zorder=3, label="точка встречи")
    ax.plot([0], [0], "o", ms=13, color="#fbbc04", label="Солнце (звезда)")

    jupiter_trail, = ax.plot([], [], color="#34a853", lw=2.5, alpha=0.35, label="путь Юпитера по орбите")
    line_t, = ax.plot([], [], color="#ea4335", lw=1.9, label="межпланетный перелёт Земля → Юпитер")
    line_p, = ax.plot([], [], color="#e37400", lw=2.3, label="траектория ухода из системы!")

    earth_dot, = ax.plot([], [], "o", ms=6.5, color="#1a73e8", zorder=4, label="Земля")
    jupiter_dot, = ax.plot([], [], "o", ms=10.5, color="#34a853", mec="#1b5e20", mew=1.1, zorder=5, label="Юпитер")
    ship, = ax.plot([], [], "o", ms=8.5, color="white", mec="#202124", mew=1.5, zorder=6, label="КА")
    ax.legend(loc="upper right", fontsize=7.0)

    # --- Врезка «Система Земли» (фаза 0) -------------------------------------------------
    ax_e = fig.add_axes([0.70, 0.56, 0.28, 0.36])
    ax_e.set_aspect("equal", adjustable="box")
    ax_e.set_xlim(-5.6, 5.6)
    ax_e.set_ylim(2.4, 10.6)
    ax_e.grid(False)
    ax_e.set_title("Система Земли: вывод на орбиту 100 км (h ×20)", fontsize=9.2)
    ax_e.tick_params(labelsize=7.5)
    ax_e.set_xlabel("x, тыс. км", fontsize=8)
    ax_e.set_ylabel("y, тыс. км", fontsize=8)
    r_ring = (R_EARTH + EXAG * V_ORBIT_TARGET) / 1e6
    ax_e.add_patch(Circle((0, 0), R_EARTH / 1e6, color="#1a73e8", alpha=0.9, zorder=2))
    ax_e.add_patch(Circle((0, 0), r_ring, fill=False, ls="--", color="#5f6368", lw=1.2, zorder=3))
    ax_e.text(0.0, 4.3, "Земля", fontsize=8.5, color="white", ha="center", zorder=4)
    ax_e.text(r_ring * 0.40, r_ring * 0.98, "орбита 100 км", fontsize=7.5,
              color="#5f6368", ha="center", zorder=4)
    e_path, = ax_e.plot([], [], color="#e37400", lw=3.0, zorder=5)
    e_sep, = ax_e.plot([], [], "x", ms=7, color="#5f6368", mew=1.6, zorder=6)
    e_ship, = ax_e.plot([], [], "o", ms=6.0, color="#ea4335", mec="white", mew=1.2, zorder=7)

    # --- Врезка «Система Юпитера» (фазы 1–3) ---------------------------------------------
    ax_j = fig.add_axes([0.70, 0.56, 0.28, 0.36])
    ax_j.set_aspect("equal", adjustable="box")
    ax_j.set_xlim(-700.0, 700.0)
    ax_j.set_ylim(-700.0, 700.0)
    ax_j.grid(False)
    ax_j.set_title("Система Юпитера (h_p = 200 км, δ = 158°)", fontsize=9.2)
    ax_j.tick_params(labelsize=7.5)
    ax_j.set_xlabel("x, тыс. км", fontsize=8)
    ax_j.set_ylabel("y, тыс. км", fontsize=8)
    ax_j.add_patch(Circle((0, 0), R_JUPITER / 1e6, color="#34a853", alpha=0.9, zorder=3))
    v_arrow = ax_j.annotate("", xy=(0, -330), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.6))
    v_text = ax_j.text(28, -320, "v_Юпитера", fontsize=8, color="#1b5e20")
    ax_j.plot(sc["loc_fly"][:, 0] / 1e6, sc["loc_fly"][:, 1] / 1e6,
              color="#e37400", lw=1.0, alpha=0.30, ls="--", zorder=1)
    zline, = ax_j.plot([], [], color="#e37400", lw=2.0, zorder=4)
    zship, = ax_j.plot([], [], "o", ms=8.5, color="#ea4335", mec="white", mew=1.2, zorder=6)
    burn_flash, = ax_j.plot([], [], "*", ms=15, color="#fbbc04", mec="#e37400", zorder=7)

    base_e = _alpha_base(ax_e)
    base_j = _alpha_base(ax_j)

    txt = fig.text(0.70, 0.50, "", va="top", fontsize=9.8, family="monospace")
    phase_txt = fig.text(0.70, 0.045, "", fontsize=9.8, weight="bold")
    bar = fig.add_axes([0.055, 0.055, 0.60, 0.015])
    bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, color="#e8eaed"))
    prog = bar.add_patch(plt.Rectangle((0, 0), 0, 1, color="#1a73e8"))

    # Камера левого плана: полная система <-> наезд на Юпитер во время пролёта
    FULL_CX, FULL_CY, FULL_HW = 0.0, -280.0, 2350.0
    jx, jy = -A_JUPITER / 1e9, 0.0
    ZOOM_HW = 220.0
    cam_on0, cam_on1 = T0 + 0.75 * T1, T0 + T1 + 0.15 * T2
    cam_off0, cam_off1 = T0 + T1 + 0.85 * T2, T0 + T1 + T2 + 0.25 * T3

    def camera(t_scr: float):
        if t_scr < cam_on0:
            p = 0.0
        elif t_scr < cam_on1:
            p = sstep((t_scr - cam_on0) / (cam_on1 - cam_on0))
        elif t_scr < cam_off0:
            p = 1.0
        elif t_scr < cam_off1:
            p = 1.0 - sstep((t_scr - cam_off0) / (cam_off1 - cam_off0))
        else:
            p = 0.0
        cx = lerp(FULL_CX, jx, p)
        cy = lerp(FULL_CY, jy, p)
        hw = lerp(FULL_HW, ZOOM_HW, p)
        return cx - hw, cx + hw, cy - hw / MAIN_ASPECT, cy + hw / MAIN_ASPECT, p

    EVENT_RU = {
        "liftoff": "взлёт",
        "cutoff Ступень 1": "сгорание ступени 1",
        "отрыв ступени 1": "отрыв ступени 1",
        "cutoff Ступень 2": "выведение на орбиту",
    }
    jupiter_history = []
    n_frames = fps * seconds

    def frame(i: int):
        t_scr = TOTAL * i / max(1, n_frames - 1)
        x0, x1, y0, y1, cam_p = camera(t_scr)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)

        if t_scr < T0:
            # --- Фаза 0: старт с Земли -> орбита100 км (врезка «Система Земли») ---
            u = t_scr / T0
            full_trail = np.vstack([sc["asc_path"], sc["coast_path"]])
            n_asc = len(sc["asc_path"])
            if u <= 0.8:
                k = int(round((u / 0.8) * (n_asc - 1)))
                k = min(k, n_asc - 1)
                cur = sc["asc_pts"][k]
            else:
                kc = int(round(((u - 0.8) / 0.2) * (len(sc["coast_path"]) - 1)))
                kc = min(kc, len(sc["coast_path"]) - 1)
                k = n_asc - 1 + kc
                p_c = sc["coast_path"][kc]
                cur = dict(x=p_c[0], y=p_c[1], t=asc["t_final"], h=asc["h_final"],
                           v=asc["v_final"], m=asc["m_final"],
                           gamma_deg=asc["gamma_final_deg"])
            e_path.set_data(full_trail[:k + 1, 0], full_trail[:k + 1, 1])
            e_ship.set_data([cur["x"]], [cur["y"]])
            if k >= sc["sep_idx"]:
                sp = sc["asc_pts"][sc["sep_idx"]]
                e_sep.set_data([sp["x"]], [sp["y"]])
            else:
                e_sep.set_data([], [])
            t_now = cur["t"]
            ev = "взлёт" if t_now < 2.0 else "ступень 1 горит"
            for e in asc["events"]:
                if e["t"] <= t_now + 1e-9 and e["event"] != "liftoff":
                    ev = EVENT_RU.get(e["event"], e["event"])
            if u > 0.8:
                ev = "выведено: орбита 100 км"
            info = (
                f"время старта   : {t_now:6.1f} с\n"
                f"высота         : {cur['h']/1e3:6.1f} км\n"
                f"скорость       : {cur['v']:6.0f} м/с\n"
                f"масса          : {cur['m']:8.0f} кг\n"
                f"программа γ    : {cur['gamma_deg']:5.1f}°\n"
                f"событие        : {ev}"
            )
            phase = ("Фаза 0: старт с Земли → орбита 100 км\n"
                     "(выведение и отбрасывание ступеней)")
            t_mis0 = -(asc["t_final"] + 1800.0) + u * asc["t_final"]
            pos_k, pos_j, ang_j = planet_positions(t_mis0, sc["tr"].tof)
            pos = pos_k.copy()
            line_t.set_data([], [])
            line_p.set_data([], [])
            burn_flash.set_data([], [])
            zline.set_data([], [])
            zship.set_data([], [])
            a_e, a_j = 1.0, 0.0
            if t_scr > T0 - 0.4:
                a_e = 1.0 - sstep((t_scr - (T0 - 0.4)) / 0.5)
        elif t_scr < T0 + T1:
            # --- Фаза 1: межпланетный перелёт к точке встречи ---
            u = (t_scr - T0) / T1
            k = int(u * (len(sc["t_tr"]) - 1))
            t_mis = float(sc["t_tr"][k])
            pos = sc["path_tr"][k]
            line_t.set_data(sc["path_tr"][:k + 1, 0] / 1e9, sc["path_tr"][:k + 1, 1] / 1e9)
            line_p.set_data([], [])
            pos_k, pos_j, ang_j = planet_positions(t_mis, sc["tr"].tof)
            burn_flash.set_data([], [])
            d_meet = float(np.linalg.norm(pos - sc["r_meet"])) / 1e9
            phase = ("Фаза 1: перелёт Земля → Юпитер\n"
                     "(аппарат летит к точке встречи)")
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"долгота Юпитера : {math.degrees(ang_j)%360:7.1f}°\n"
                f"до точки встречи: {d_meet:7.2f} млн км\n"
                f"Δv старта     : {sc['tr'].dv_tmi:7.0f} м/с\n"
                f"v прилёта     : {sc['tr'].v_arr:7.0f} м/с"
            )
            a_j = sstep((t_scr - T0) / 0.6)          # появление кадра «Система Юпитера»
            a_e = max(0.0, 1.0 - sstep((t_scr - (T0 - 0.4)) / 0.5))
        elif t_scr < T0 + T1 + T2:
            # --- Фаза 2: пролёт Юпитера (замедление у перицентра, наезд камеры) ---
            u = (t_scr - T0 - T1) / T2
            # Время: замедление у перицентра (степенной маппинг: скорость ~0 при u=0.5)
            w = 2.0 * u - 1.0
            t_off = sc["t_half"] * math.copysign(abs(w) ** 1.75, w)
            k = int(np.searchsorted(sc["dt_fly"], t_off))
            k = min(max(k, 0), len(sc["t_fly"]) - 1)
            t_mis = float(sc["t_fly"][k])
            pos = sc["path_fly"][k]
            loc = sc["loc_fly"][k] / 1e6
            zline.set_data(sc["loc_fly"][:k + 1, 0] / 1e6, sc["loc_fly"][:k + 1, 1] / 1e6)
            zship.set_data([loc[0]], [loc[1]])
            if abs(t_off) < 0.14 * DAY:
                burn_flash.set_data([loc[0]], [loc[1]])
            else:
                burn_flash.set_data([], [])
            pos_k, pos_j, ang_j = planet_positions(t_mis, sc["tr"].tof)
            h_now = (np.linalg.norm(sc["loc_fly"][k]) - R_JUPITER) / 1e3
            phase = ("Фаза 2: пролёт Юпитера — крупный план\n"
                     f"(замедление, Оберт {sc['dv_oberth']:.1f} м/с)")
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"высота над Юпит.: {h_now:7.0f} км\n"
                f"перицентр h_p : {sc['h_p_km']:7.0f} км\n"
                f"поворот δ     : {sc['delta_deg']:7.2f}°\n"
                f"импульс Оберта: {sc['dv_oberth']:7.1f} м/с"
            )
            a_e, a_j = 0.0, 1.0
            # Масштаб врезки следует за КА: крупный план у перицентра (>=280),
            # отъезд/приближение расширяют кадр (<=700), КА не покидает вид
            r_now_loc = float(np.linalg.norm(sc["loc_fly"][k])) / 1e6   # тыс. км
            hw = float(np.clip(1.45 * r_now_loc, 280.0, 700.0))
            ax_j.set_xlim(-hw, hw)
            ax_j.set_ylim(-hw, hw)
            v_arrow.xy = (0.0, -0.47 * hw)
            v_text.set_position((0.04 * hw, -0.44 * hw))
        else:
            # --- Фаза 3: уход из Солнечной системы ---
            u = (t_scr - T0 - T1 - T2) / T3
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
            phase = "Фаза 3: УХОД ИЗ СОЛНЕЧНОЙ СИСТЕМЫ!\n(v > v_esc: КА покидает систему)"
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"расст. от зв. : {r_now/1e9:7.1f} млн км\n"
                f"скорость КА   : {v_now:7.0f} м/с\n"
                f"2-я косм. зв. : {v_loc_esc:7.0f} м/с\n"
                f"статус: УХОД ИЗ СИСТЕМЫ (v > v_esc)"
            )
            a_e, a_j = 0.0, 1.0

        set_ax_alpha(ax_e, base_e, a_e)
        set_ax_alpha(ax_j, base_j, a_j)

        jupiter_history.append(pos_j / 1e9)
        jh = np.array(jupiter_history)
        jupiter_trail.set_data(jh[:, 0], jh[:, 1])

        earth_dot.set_data([pos_k[0] / 1e9], [pos_k[1] / 1e9])
        jupiter_dot.set_data([pos_j[0] / 1e9], [pos_j[1] / 1e9])
        # При наезде камеры на Юпитер КА виден только в врезке (не сливается с планетой)
        if cam_p < 0.3:
            ship.set_data([pos[0] / 1e9], [pos[1] / 1e9])
        else:
            ship.set_data([], [])
        prog.set_width(i / max(1, n_frames - 1))
        phase_txt.set_text(phase)
        txt.set_text(info)
        return []

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2600,
                          metadata={"title": "Earth Launch to Jupiter Flyby & System Escape"})
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Видео сохранено: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "mission.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=22)
    args = ap.parse_args()
    make(args.out, fps=args.fps, seconds=args.seconds)
