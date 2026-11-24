"""
Анимация миссии «Вояджер в системе Кербола» для видеоотчёта.

Сценарий (16 с, 30 кадров/с):
  1) гомановский перелёт Кербин → Юл           (0–6 с, время сжато);
  2) пролёт Юла, замедленная съёмка вблизи     (6–11 с, врезка с гиперболой);
  3) послепролётная орбита к афелию Эилу       (11–16 с).

Траектории строятся аналитически (конические сечения + уравнение Кеплера),
поэтому кадры не зависят от численного интегратора и анимация идёт быстро.

Запуск:  python3 anim.py [--out out/mission.mp4] [--fps 30] [--preview]
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import matplotlib

# matplotlib ищет ffmpeg в PATH — подкладываем бинарник из imageio-ffmpeg (если есть)
try:
    import imageio_ffmpeg

    os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ["PATH"]
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from sim import (MU_SUN, MU_JOOL, R_JOOL, A_JOOL, SOI_JOOL, A_KERBIN,
                 launch_window, delta_hyp, b_hyp, helio_elements, post_flyby,
                 eeloo_pos, OUT)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True})

DAY = 86_400.0


# --- кинематика конических сечений ---------------------------------------------------

def kepler_E(M, e, iters=40):
    """Решение уравнения Кеплера M = E − e sin E методом Ньютона (эллипс)."""
    E = M if e < 0.8 else np.pi * np.ones_like(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def ellipse_state(a, e, t, mu=MU_SUN, omega=0.0):
    """Положение на эллипсе в момент t от перицентра (плоско 2D)."""
    n = math.sqrt(mu / a ** 3)
    M = np.mod(n * t, 2 * np.pi)
    E = kepler_E(M, e)
    x = a * (np.cos(E) - e)
    y = a * math.sqrt(1 - e ** 2) * np.sin(E)
    c, s = math.cos(omega), math.sin(omega)
    return np.array([c * x - s * y, s * x + c * y])


def hyperbola_state(rp, e, t, mu=MU_JOOL):
    """
    Положение на гиперболе (2D, перицентр на оси +x при t=0) по времени t.
    Гиперболическое уравнение Кеплера: M = e·sh H − H, t = M / n.
    """
    a = rp / (e - 1)                     # |a| гиперболы
    n = math.sqrt(mu / abs(a) ** 3)
    M = n * t
    H = math.asinh(M / max(e, 1.0001))
    for _ in range(60):
        H = H - (e * math.sinh(H) - H - M) / (e * math.cosh(H) - 1)
    x = a * (e - math.cosh(H))
    y = a * math.sqrt(e ** 2 - 1) * math.sinh(H)
    return np.array([x, y])


# --- геометрия миссии ----------------------------------------------------------------

def build_scene(rp_km=300.0, n_transfer=1200, n_flyby=2400, n_post=1600):
    tr = launch_window()
    rp = R_JOOL + rp_km * 1e3

    # 1. перелёт: эллипс от Кербина (перигелий) до Юла (афелий)
    a_t = tr.a_t
    e_t = (A_JOOL - A_KERBIN) / (A_JOOL + A_KERBIN)
    t_transfer = np.linspace(0, tr.tof, n_transfer)
    path_t = np.array([ellipse_state(a_t, e_t, t) for t in t_transfer])

    # 2. пролёт: гипербола в системе Юла; Юл движется по круговой орбите
    e_h = 1 + rp * tr.v_inf_j ** 2 / MU_JOOL
    omega_j = math.sqrt(MU_SUN / A_JOOL ** 3)
    t_inf = 2.2 * DAY
    t_fly = np.linspace(-t_inf, t_inf, n_flyby)
    # входящая асимптота в локальных координатах направлена вдоль +x;
    # поворачиваем так, чтобы относительная скорость была против движения Юла
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])
    jool0 = np.array([A_JOOL, 0.0])
    path_f, v_rel = [], []
    for t in t_fly:
        loc = hyperbola_state(rp, e_h, t)
        loc = R @ loc
        ang = omega_j * t
        rot = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        path_f.append(rot @ (jool0 + loc))
    path_f = np.array(path_f)

    # 3. послепролёта: эллипс от точки Юла
    vinf_out, v_out = post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)
    el = helio_elements(np.array([A_JOOL, 0.0]), v_out)
    a2, e2 = el["a"], el["e"]
    t_post = np.linspace(0, 0.75 * 2 * math.pi * math.sqrt(a2 ** 3 / MU_SUN), n_post)
    path_p = np.array([ellipse_state(a2, e2, t) for t in t_post])

    return dict(tr=tr, rp=rp, e_h=e_h, path_t=path_t, path_f=path_f, path_p=path_p,
                t_transfer=t_transfer, t_fly=t_fly, t_post=t_post,
                a2=a2, e2=e2, omega_j=omega_j, jool0=jool0,
                v_inf_out=vinf_out, v_out=v_out, delta=math.degrees(delta_hyp(rp, tr.v_inf_j)))


# --- сама анимация -------------------------------------------------------------------

def make(out_path: str, fps: int = 30, seconds: int = 16):
    sc = build_scene()
    R = np.array([[0.0, 1.0], [-1.0, 0.0]])
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_axes([0.055, 0.13, 0.60, 0.80])
    ax.set_aspect("equal")
    lim = 1.75e11
    ax.set_xlim(-1.25e11 / 1e9, 1.6e11 / 1e9)
    ax.set_ylim(-1.35e11 / 1e9, 1.35e11 / 1e9)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Миссия «Вояджер» в системе Кербола — гравитационный манёвр у Юла")

    th = np.linspace(0, 2 * np.pi, 600)
    ax.plot(A_KERBIN * np.cos(th) / 1e9, A_KERBIN * np.sin(th) / 1e9,
            color="#8ab4f8", lw=1.1, label="орбита Кербина")
    ax.plot(A_JOOL * np.cos(th) / 1e9, A_JOOL * np.sin(th) / 1e9,
            color="#34a853", lw=1.1, label="орбита Юла")
    ax.plot([0], [0], "o", ms=14, color="#fbbc04", label="Кербол")

    # следы траекторий
    line_t, = ax.plot([], [], color="#ea4335", lw=1.8, alpha=0.9, label="перелёт")
    line_f, = ax.plot([], [], color="#9334e6", lw=1.6, alpha=0.9, label="пролёт Юла")
    line_p, = ax.plot([], [], color="#e37400", lw=1.8, alpha=0.9, label="после пролёта")
    ship, = ax.plot([], [], "o", ms=10, color="white", mec="#202124", mew=1.4, zorder=5)
    jool_dot, = ax.plot([], [], "o", ms=11, color="#34a853", zorder=4)
    ax.legend(loc="upper left", fontsize=8)

    # врезка: крупный план Юла
    axz = fig.add_axes([0.70, 0.58, 0.28, 0.34])
    axz.set_aspect("equal", adjustable="box")
    axz.autoscale(False)
    axz.set_title("крупный план: гипербола у Юла", fontsize=9)
    axz.tick_params(labelsize=7)
    axz.set_xlabel("x, тыс. км", fontsize=8)
    axz.set_ylabel("y, тыс. км", fontsize=8)
    zoom_kkm = 250.0                          # ±250 тыс. км
    axz.set_xlim(-zoom_kkm, zoom_kkm)
    axz.set_ylim(-zoom_kkm, zoom_kkm)
    axz.add_patch(plt.Circle((0, 0), R_JOOL / 1e6, color="#34a853", alpha=0.9, zorder=2))
    # «призрак» всей аналитической гиперболы — чтобы было видно форму траектории
    ghost_loc = np.array([R @ hyperbola_state(sc["rp"], sc["e_h"], t) for t in sc["t_fly"]])
    axz.plot(ghost_loc[:, 0] / 1e6, ghost_loc[:, 1] / 1e6,
             color="#9334e6", lw=0.9, alpha=0.28, ls="--", zorder=1)
    zline, = axz.plot([], [], color="#9334e6", lw=1.8, zorder=3)
    zship, = axz.plot([], [], "o", ms=9, color="#ea4335", mec="white", mew=1.2, zorder=6)
    zpath = []
    axz.set_xlim(-zoom_kkm, zoom_kkm)
    axz.set_ylim(-zoom_kkm, zoom_kkm)

    # панель телеметрии
    txt = fig.text(0.70, 0.52, "", va="top", fontsize=10.5, family="monospace")
    phase_txt = fig.text(0.70, 0.04, "", fontsize=10.5, weight="bold", wrap=True)
    bar = fig.add_axes([0.055, 0.055, 0.60, 0.014])
    bar.set_xlim(0, 1); bar.set_ylim(0, 1); bar.axis("off")
    bar.add_patch(plt.Rectangle((0, 0), 1, 1, color="#e8eaed"))
    prog = bar.add_patch(plt.Rectangle((0, 0), 0, 1, color="#1a73e8"))

    # таймлайн: 6 с перелёт, 5 с пролёт, 5 с послепролёта
    T1, T2, T3 = 6.0, 5.0, 5.0
    total = T1 + T2 + T3

    def frame(i):
        t_scr = total * i / (fps * seconds)
        if t_scr <= T1:            # перелёт
            u = t_scr / T1
            k = int(u * (len(sc["path_t"]) - 1))
            pos = sc["path_t"][k]
            t_mission = sc["t_transfer"][k]
            line_t.set_data(sc["path_t"][:k + 1, 0] / 1e9, sc["path_t"][:k + 1, 1] / 1e9)
            jpos = np.array([A_JOOL, 0.0])
            phase = "Фаза 1 — гомановский перелёт Кербин → Юл"
            info = (f"время миссии  : {t_mission/DAY:7.1f} сут\n"
                    f"расстояние    : {np.linalg.norm(pos)/1e9:7.2f} млн км\n"
                    f"v∞ у Кербина  : {sc['tr'].v_inf_k:7.0f} м/с\n"
                    f"Δv отлёта     : {sc['tr'].dv_tmi:7.0f} м/с")
            zoom_out = True
        elif t_scr <= T1 + T2:     # пролёт
            u = (t_scr - T1) / T2
            k = int(u * (len(sc["path_f"]) - 1))
            pos = sc["path_f"][k]
            t_mission = sc["tr"].tof + sc["t_fly"][k]
            line_f.set_data(sc["path_f"][:k + 1, 0] / 1e9, sc["path_f"][:k + 1, 1] / 1e9)
            ang = sc["omega_j"] * sc["t_fly"][k]
            jpos = np.array([A_JOOL * math.cos(ang), A_JOOL * math.sin(ang)])
            # крупный план
            loc = (pos - jpos) / 1e6                  # тыс. км от центра Юла
            zpath.append(loc)
            zp = np.array(zpath)
            zline.set_data(zp[:, 0], zp[:, 1])
            zship.set_data([loc[0]], [loc[1]])
            phase = "Фаза 2 — пролёт Юла: поворот траектории гравитацией"
            dv = math.degrees(delta_hyp(sc["rp"], sc["tr"].v_inf_j))
            info = (f"время миссии  : {t_mission/DAY:7.1f} сут\n"
                    f"высота над Юлом: {(np.linalg.norm(pos-jpos)-R_JOOL)/1e3:8.0f} км\n"
                    f"v∞            : {sc['tr'].v_inf_j:7.0f} м/с\n"
                    f"δ (полный)    : {dv:7.1f}°")
            zoom_out = False
        else:                      # после пролёта
            u = (t_scr - T1 - T2) / T3
            k = int(u * (len(sc["path_p"]) - 1))
            pos = sc["path_p"][k]
            t_mission = sc["tr"].tof + sc["t_fly"][-1] + sc["t_post"][k]
            line_p.set_data(sc["path_p"][:k + 1, 0] / 1e9, sc["path_p"][:k + 1, 1] / 1e9)
            ang = sc["omega_j"] * sc["t_fly"][-1]
            jpos = np.array([A_JOOL * math.cos(ang), A_JOOL * math.sin(ang)])
            phase = "Фаза 3 — после пролёта: афелий в 10 раз дальше орбиты Юла"
            info = (f"время миссии  : {t_mission/DAY:7.1f} сут\n"
                    f"расстояние    : {np.linalg.norm(pos)/1e9:7.2f} млн км\n"
                    f"афелий        : {sc['a2']*(1+sc['e2'])/1e9:7.1f} млн км\n"
                    f"Δv            : 0 м/с (только гравитация)")
            zoom_out = False

        ship.set_data([pos[0] / 1e9], [pos[1] / 1e9])
        jool_dot.set_data([jpos[0] / 1e9], [jpos[1] / 1e9])
        prog.set_width(i / (fps * seconds))
        phase_txt.set_text(phase)
        txt.set_text(info)
        return []

    anim = FuncAnimation(fig, frame, frames=fps * seconds, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2600,
                          metadata={"title": "Voyager-K flyby animation"})
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)
    print(f"Готово: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT, "mission.mp4"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=int, default=16)
    args = ap.parse_args()
    make(args.out, fps=args.fps, seconds=args.seconds)
