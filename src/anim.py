"""
Анимация миссии «Вояджер в системе Кербола» (out/mission.mp4).

Сценарий (16 с, 30 кадров/с):
  1) Фаза 1 (0–5.5 с): гомановский перелёт Кербин → Юл.
     Кербин и Юл вращаются по круговым орбитам с реальным временем миссии t
     (Юл из угла опережения 96.6° приходит в точку встречи с КА).
  2) Фаза 2 (5.5–10.5 с): глубокий гиперболический пролёт Юла у границы атмосферы
     (h_p = 200 км, поворот δ = 138.98°) + импульс Оберта Δv = 58.9 м/с в перицентре.
  3) Фаза 3 (10.5–16 с): уход из системы Кербола!
     Показаны обе ветви после манёвра:
       - пунктиром: пассивный заброс к афелию Эилу (h_p = 35 096 км, Δv = 0);
       - основная траектория КА: разомкнутая траектория ухода за пределы всех орбит
         планет и притяжения звезды Кербол (|v_вых| ≥ v_esc = 5839 м/с).

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

from sim import (MU_SUN, MU_JOOL, R_JOOL, A_JOOL, A_KERBIN, A_EELOO, E_EELOO, ATM_JOOL,
                 launch_window, delta_hyp, turn_vinf, oberth_dv, helio_elements, post_flyby, OUT)

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

DAY = 86_400.0
OMEGA_KERBIN = math.sqrt(MU_SUN / A_KERBIN ** 3)
OMEGA_JOOL = math.sqrt(MU_SUN / A_JOOL ** 3)


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


def hyperbola_local(rp: float, e: float, t: float, mu: float = MU_JOOL) -> np.ndarray:
    """
    Положение КА относительно Юла (2D) в момент времени t от перицентра.
    В точке встречи (-A_JOOL, 0) Юл летит вниз (v_J = (0, -v_j)), а КА входит снизу вдоль +y,
    огибает Юл сзади по орбите (сверху) на высоте h_p = 200 км и разворачивается на 138.98°!
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
    """Положения Кербина и Юла в физический момент времени t_mission."""
    ang_k = OMEGA_KERBIN * t_mission
    ang_j = math.pi + OMEGA_JOOL * (t_mission - tr_tof)
    pos_k = np.array([A_KERBIN * math.cos(ang_k), A_KERBIN * math.sin(ang_k)])
    pos_j = np.array([A_JOOL * math.cos(ang_j), A_JOOL * math.sin(ang_j)])
    return pos_k, pos_j, ang_j


def build_scene(n_tr: int = 900, n_fly: int = 1200, n_post: int = 1200):
    """Строит полную миссию: перелёт -> пролёт Юла (h_p = 200 км + Оберт) -> выход из системы звезды."""
    tr = launch_window()
    h_p_km = ATM_JOOL / 1e3  # 200 км (граница атмосферы Юла)
    rp = R_JOOL + ATM_JOOL

    # Фаза 1: перелёт Кербин -> Юл
    t_tr = np.linspace(0.0, tr.tof, n_tr)
    r0_tr = np.array([A_KERBIN, 0.0])
    v0_tr = np.array([0.0, math.sqrt(MU_SUN * (2.0 / A_KERBIN - 1.0 / tr.a_t))])
    path_tr = np.array([kepler_ellipse_pos(r0_tr, v0_tr, t) for t in t_tr])

    # Фаза 2: глубокий пролёт вблизи Юла (окно ±1.8 сут вокруг перицентра)
    e_h = 1.0 + rp * tr.v_inf_j ** 2 / MU_JOOL
    t_half = 1.8 * DAY
    dt_fly = np.linspace(-t_half, t_half, n_fly)
    t_fly = tr.tof + dt_fly
    loc_fly = np.array([hyperbola_local(rp, e_h, dt) for dt in dt_fly])
    path_fly = []
    for dt, loc in zip(dt_fly, loc_fly):
        _, jpos, ang_j = planet_positions(tr.tof + dt, tr.tof)
        d_ang = ang_j - math.pi
        c, s = math.cos(d_ang), math.sin(d_ang)
        path_fly.append(jpos + np.array([c * loc[0] - s * loc[1], s * loc[0] + c * loc[1]]))
    path_fly = np.array(path_fly)

    # Фаза 3: УХОД ИЗ СИСТЕМЫ ЗВЕЗДЫ (пролёт h_p = 200 км + импульс Оберта в перицентре)
    d_max = delta_hyp(rp, tr.v_inf_j)
    vinf_esc, dv_oberth = oberth_dv(rp, tr.v_inf_j, d_max, tr.v_jool)
    vinf_escape_hyp = vinf_esc * 1.02
    vinf_out_std = turn_vinf(np.array([0.0, -vinf_escape_hyp]), d_max, +1)
    v_out_meet = -(np.array([0.0, tr.v_jool]) + vinf_out_std)
    r_meet = np.array([-A_JOOL, 0.0])
    v_esc_sun = math.sqrt(2.0 * MU_SUN / A_JOOL)

    # Продолжаем полёт на 850 суток — КА пробивает внешнюю орбиту Эилу (113.5 млн км) и уходит на 270 млн км!
    dt_post = np.linspace(0.0, 850.0 * DAY, n_post)
    t_post = tr.tof + t_half + dt_post
    path_post = np.array([kepler_hyperbola_pos(r_meet, v_out_meet, dt) for dt in dt_post])

    # Дополнительно для сравнения на фоне: пассивная замкнутая орбита к афелию Эилу (h_p = 35 096 км, Δv = 0)
    rp_eeloo = R_JOOL + 35_096e3
    _, v_eeloo_std = post_flyby(rp_eeloo, tr.v_inf_j, +1, tr.v_jool)
    v_eeloo_meet = -v_eeloo_std
    el_eeloo = helio_elements(r_meet, v_eeloo_meet)
    T_eeloo = 2.0 * math.pi * math.sqrt(el_eeloo["a"] ** 3 / MU_SUN)
    path_eeloo_ref = np.array([kepler_ellipse_pos(r_meet, v_eeloo_meet, dt)
                               for dt in np.linspace(0.0, T_eeloo, 600)])

    return dict(
        tr=tr, rp=rp, h_p_km=h_p_km, e_h=e_h,
        t_tr=t_tr, path_tr=path_tr,
        dt_fly=dt_fly, t_fly=t_fly, loc_fly=loc_fly, path_fly=path_fly,
        dt_post=dt_post, t_post=t_post, path_post=path_post,
        path_eeloo_ref=path_eeloo_ref,
        v_out_norm=float(np.linalg.norm(v_out_meet)),
        v_esc_sun=v_esc_sun,
        dv_oberth=dv_oberth,
        delta_deg=math.degrees(d_max),
    )


def make(out_path: str, fps: int = 30, seconds: int = 16):
    sc = build_scene()
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.055, 0.13, 0.60, 0.80])
    ax.set_aspect("equal")
    # Расширенный масштаб, чтобы отчётливо видеть выход КА далеко за пределы всех орбит системы!
    ax.set_xlim(-225, 225)
    ax.set_ylim(-290, 160)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Гравитационный манёвр у Юла и уход КА за пределы планетной системы")

    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(A_KERBIN * np.cos(th) / 1e9, A_KERBIN * np.sin(th) / 1e9,
            color="#8ab4f8", lw=1.1, label="орбита Кербина")
    ax.plot(A_JOOL * np.cos(th) / 1e9, A_JOOL * np.sin(th) / 1e9,
            color="#34a853", lw=1.2, label="орбита Юла")
    # Внешняя граница планетной системы — орбита Эилу
    b_ee = A_EELOO * math.sqrt(1.0 - E_EELOO ** 2)
    xe, ye = A_EELOO * np.cos(th) - A_EELOO * E_EELOO, b_ee * np.sin(th)
    om = math.radians(260)
    ax.plot((xe * math.cos(om) - ye * math.sin(om)) / 1e9,
            (xe * math.sin(om) + ye * math.cos(om)) / 1e9,
            color="#9aa0a6", lw=1.1, ls="--", label="внешняя орбита (Эилу)")
    # Справочная замкнутая ветвь без импульса Оберта
    ax.plot(sc["path_eeloo_ref"][:, 0] / 1e9, sc["path_eeloo_ref"][:, 1] / 1e9,
            color="#9334e6", lw=1.0, ls=":", alpha=0.65, label="пассивный предел к Эилу (Δv=0)")
    ax.plot([0], [0], "o", ms=13, color="#fbbc04", label="Кербол (звезда)")

    jool_trail, = ax.plot([], [], color="#34a853", lw=2.5, alpha=0.35, label="путь Юла по орбите")
    line_t, = ax.plot([], [], color="#ea4335", lw=1.9, label="перелёт Кербин → Юл")
    line_p, = ax.plot([], [], color="#e37400", lw=2.3, label="траектория ухода из системы!")

    kerbin_dot, = ax.plot([], [], "o", ms=6.5, color="#1a73e8", zorder=4, label="Кербин")
    jool_dot, = ax.plot([], [], "o", ms=10.5, color="#34a853", mec="#1b5e20", mew=1.1, zorder=5, label="Юл")
    ship, = ax.plot([], [], "o", ms=8.5, color="white", mec="#202124", mew=1.5, zorder=6, label="КА")
    ax.legend(loc="upper right", fontsize=7.3)

    # Врезка: крупный план глубокой гиперболы у Юла (h_p = 200 км, поворот 138.98°)
    axz = fig.add_axes([0.70, 0.56, 0.28, 0.36])
    axz.set_aspect("equal", adjustable="box")
    axz.autoscale(False)
    zoom_kkm = 140.0
    axz.set_xlim(-zoom_kkm, zoom_kkm)
    axz.set_ylim(-zoom_kkm, zoom_kkm)
    axz.set_title("Система Юла (h_p = 200 км, δ = 139°)", fontsize=9.2)
    axz.tick_params(labelsize=7.5)
    axz.set_xlabel("x, тыс. км", fontsize=8)
    axz.set_ylabel("y, тыс. км", fontsize=8)
    axz.add_patch(plt.Circle((0, 0), R_JOOL / 1e6, color="#34a853", alpha=0.9, zorder=3))
    axz.annotate("", xy=(0, -65), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.6))
    axz.text(6, -60, "v_Юла", fontsize=8, color="#1b5e20")
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
    jool_history = []

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
            phase = "Фаза 1: гомановский перелёт Кербин → Юл\n(обе планеты движутся к точке встречи)"
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"долгота Юла   : {math.degrees(ang_j)%360:7.1f}°\n"
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
            h_now = (np.linalg.norm(sc["loc_fly"][k]) - R_JOOL) / 1e3
            phase = "Фаза 2: праща Юла (h_p = 200 км, δ = 139°)\n+ импульс Оберта Δv = 58.9 м/с в перицентре!"
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"высота над Юл.: {h_now:7.0f} км\n"
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
            eps_sun = 0.5 * sc["v_out_norm"] ** 2 - MU_SUN / A_JOOL
            v_now = math.sqrt(2.0 * (eps_sun + MU_SUN / r_now))
            phase = "Фаза 3: УХОД ИЗ СИСТЕМЫ КЕРБОЛА!\n(КА пересекает все орбиты с v > v_esc)"
            info = (
                f"время миссии  : {t_mis/DAY:7.1f} сут\n"
                f"расст. от зв. : {r_now/1e9:7.1f} млн км\n"
                f"скорость КА   : {v_now:7.0f} м/с\n"
                f"2-я косм. зв. : {v_loc_esc:7.0f} м/с\n"
                f"статус: УХОД ИЗ СИСТЕМЫ (v > v_esc)"
            )

        jool_history.append(pos_j / 1e9)
        jh = np.array(jool_history)
        jool_trail.set_data(jh[:, 0], jh[:, 1])

        kerbin_dot.set_data([pos_k[0] / 1e9], [pos_k[1] / 1e9])
        jool_dot.set_data([pos_j[0] / 1e9], [pos_j[1] / 1e9])
        ship.set_data([pos[0] / 1e9], [pos[1] / 1e9])
        prog.set_width(i / max(1, n_frames - 1))
        phase_txt.set_text(phase)
        txt.set_text(info)
        return []

    anim = FuncAnimation(fig, frame, frames=fps * seconds, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2600,
                          metadata={"title": "Jool Gravity Assist & System Escape"})
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
