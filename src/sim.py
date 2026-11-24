"""
Прототип физико-математической модели проекта
«Гравитационный манёвр у газового гиганта: "Вояджер" в системе Кербола».

Модель: плоская (2D) ограниченная задача трёх тел, метод точечных сфер действия.
Участки:
  1) гомановский перелёт Кербин -> Юл в поле Кербола;
  2) гиперболический пролёт в сфере действия Юла (аналитика + проверка RK4);
  3) послепролётная гелиоцентрическая орбита (энергетический метод);
  4) условие ухода из системы и роль импульса в перицентре (эффект Оберта);
  5) задача Ламберта: наведение на орбиту Эилу (Юл -> Эилу).

Единицы: СИ. Исходные данные — KSP (stock), сверены с KSP Wiki.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "axes.axisbelow": True, "savefig.bbox": "tight",
})

# Корень репозитория — каталог над src/ (результаты всегда в ./out, не в src/out)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# --- 1. Константы системы Кербола (KSP 1.12, stock) ---------------------------------
MU_SUN = 1.1723328e18
MU_KERBIN, R_KERBIN, A_KERBIN = 3.5316e12, 600_000.0, 13_599_840_256.0
MU_JOOL, R_JOOL, A_JOOL = 2.82528e14, 6_000_000.0, 68_773_560_320.0
SOI_JOOL, ATM_JOOL = 2_455_985_200.0, 200_000.0
MU_EELOO, R_EELOO, A_EELOO, E_EELOO = 7.441e10, 210_000.0, 90_118_820_000.0, 0.26
DAY, KSP_DAY = 86_400.0, 21_600.0


def period(a, mu):
    return 2 * math.pi * math.sqrt(a ** 3 / mu)


def v_circ(r, mu):
    return math.sqrt(mu / r)


def v_esc(r, mu):
    return math.sqrt(2 * mu / r)


# --- 2. Перелёт Кербин -> Юл ---------------------------------------------------------

@dataclass
class Transfer:
    r1: float
    a_t: float
    tof: float
    dv_tmi: float
    v_inf_k: float
    v_arr: float
    v_jool: float
    v_inf_j: float
    phase_deg: float


def launch_window(r1=R_KERBIN + 100_000.0) -> Transfer:
    mu = MU_SUN
    a_t = 0.5 * (A_KERBIN + A_JOOL)
    tof = math.pi * math.sqrt(a_t ** 3 / mu)
    v_c1 = v_circ(A_KERBIN, mu)
    v_inf_k = math.sqrt(mu * (2 / A_KERBIN - 1 / a_t)) - v_c1
    v_arr = math.sqrt(mu * (2 / A_JOOL - 1 / a_t))
    v_j = v_circ(A_JOOL, mu)
    v_inf_j = abs(v_j - v_arr)
    v_per = math.sqrt(v_inf_k ** 2 + 2 * MU_KERBIN / r1)
    dv = v_per - v_circ(r1, MU_KERBIN)
    n_j = 2 * math.pi / period(A_JOOL, mu)
    phase = (math.pi - n_j * tof + math.pi) % (2 * math.pi) - math.pi
    return Transfer(r1, a_t, tof, dv, v_inf_k, v_arr, v_j, v_inf_j, math.degrees(phase))


# --- 3. Гиперболический пролёт -------------------------------------------------------

def e_hyp(rp, vinf, mu=MU_JOOL):
    return 1.0 + rp * vinf ** 2 / mu


def delta_hyp(rp, vinf, mu=MU_JOOL):
    """δ = 2 arcsin(1/e) — угол поворота вектора скорости."""
    return 2 * math.asin(1.0 / e_hyp(rp, vinf, mu))


def b_hyp(rp, vinf, mu=MU_JOOL):
    e = e_hyp(rp, vinf, mu)
    return rp * math.sqrt((e + 1) / (e - 1))


def v_on_conic(rp, vinf, rr, branch, mu=MU_JOOL):
    """Скорость на гиперболе в точке радиуса rr (branch=-1 вход, +1 выход)."""
    e = e_hyp(rp, vinf, mu)
    p = rp * (1 + e)
    cosnu = (p / rr - 1) / e
    nu = branch * math.acos(max(-1.0, min(1.0, cosnu)))
    er = np.array([math.cos(nu), math.sin(nu)])
    et = np.array([-math.sin(nu), math.cos(nu)])
    gamma = math.atan2(e * math.sin(nu), 1 + e * math.cos(nu))
    return math.sqrt(vinf ** 2 + 2 * mu / rr) * (math.sin(gamma) * er + math.cos(gamma) * et)


def propagate_hyperbola(rp, vinf, mu=MU_JOOL, frac_soi=0.95, dt_max=120.0,
                        side=+1, soi=SOI_JOOL):
    """
    RK4 в инерциальной системе Юла с адаптивным шагом.
    Старт на радиусе frac_soi*SOI (модель сферы действия: вне SOI гравитация
    Юла не действует, поэтому интегрировать дальше нельзя).
    Возвращает траекторию, достигнутый перицентр и угол между входным и
    выходным векторами скорости, посчитанный на ОДНОМ и том же радиусе.
    """
    b = b_hyp(rp, vinf, mu)
    r_check = frac_soi * soi
    r = np.array([-math.sqrt(r_check ** 2 - b ** 2), side * b])
    v = np.array([vinf, 0.0])

    def acc(q):
        return -mu * q / np.dot(q, q) ** 1.5

    traj = [(r.copy(), v.copy())]
    r_min, n = np.linalg.norm(r), 0
    while True:
        rn, vn = np.linalg.norm(r), np.linalg.norm(v)
        dt = min(dt_max, max(0.5, 0.02 * rn / vn))
        a1 = acc(r); k1 = v
        a2 = acc(r + 0.5 * dt * k1); k2 = v + 0.5 * dt * a1
        a3 = acc(r + 0.5 * dt * k2); k3 = v + 0.5 * dt * a2
        a4 = acc(r + dt * k3);       k4 = v + dt * a3
        r = r + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        r_min = min(r_min, np.linalg.norm(r))
        n += 1
        if np.linalg.norm(r - traj[-1][0]) > 0.004 * np.linalg.norm(r):
            traj.append((r.copy(), v.copy()))
        if np.linalg.norm(r) > r_check and np.dot(r, v) > 0:
            break
        if n > 2_000_000:
            break
    traj.append((r.copy(), v.copy()))

    v_in = v_on_conic(rp, vinf, r_check, -1, mu)
    v_out = v_on_conic(rp, vinf, r_check, +1, mu)
    delta_fin = math.acos(np.clip(np.dot(v_in, v_out) /
                                  (np.linalg.norm(v_in) * np.linalg.norm(v_out)), -1, 1))
    return (np.array([p for p, _ in traj]), np.array([s for _, s in traj]),
            r_min, delta_fin, r_check)


# --- 4. Послепролётная гелиоцентрическая орбита (энергетический метод) ---------------

def turn_vinf(vinf_in, delta, side):
    """Поворот v∞ на угол delta (side=±1 — по/против часовой)."""
    c, s = math.cos(side * delta), math.sin(side * delta)
    R = np.array([[c, -s], [s, c]])
    return R @ np.asarray(vinf_in, dtype=float)


def post_flyby(rp, vinf, side, v_j, r_helio=A_JOOL):
    """
    До пролёта аппарат движется медленнее Юла, поэтому v∞ направлена против
    орбитального движения Юла: v∞_in = (0, -(v_j - v_arr)) = (0, -v_inf).
    """
    vinf_in = np.array([0.0, -vinf])
    vinf_out = turn_vinf(vinf_in, delta_hyp(rp, vinf), side)
    return vinf_out, np.array([0.0, v_j]) + vinf_out


def helio_elements(r_vec, v_vec, mu=MU_SUN):
    r, v = np.linalg.norm(r_vec), np.linalg.norm(v_vec)
    eps = v * v / 2 - mu / r
    h = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]
    a = -mu / (2 * eps) if eps < 0 else math.inf
    e_vec = ((v * v - mu / r) * r_vec - np.dot(r_vec, v_vec) * v_vec) / mu
    e = np.linalg.norm(e_vec)
    r_per = h * h / mu / (1 + e)
    return dict(eps=eps, a=a, e=e, r_per=r_per,
                r_apo=a * (1 + e) if math.isfinite(a) else math.inf)


def rp_for_apo(apo_target, vinf, side, v_j, r_helio=A_JOOL):
    """Перицентр пролёта, при котором афелий = apo_target (бисекция)."""
    def f(rp):
        _, v_out = post_flyby(rp, vinf, side, v_j, r_helio)
        return helio_elements(np.array([r_helio, 0.0]), v_out)["r_apo"] - apo_target
    lo, hi = R_JOOL + ATM_JOOL + 10_000, 200 * R_JOOL
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def oberth_dv(rp, vinf, delta_needed, v_j, mu=MU_JOOL):
    """Δv в перицентре, чтобы при том же δ получить нужную |v∞'|."""
    v_esc_helio = v_esc(A_JOOL, MU_SUN)
    # |v∞'|^2 - 2 v_j cos δ |v∞'| + v_j^2 - v_esc^2 = 0
    bb, cc = -2 * v_j * math.cos(delta_needed), v_j ** 2 - v_esc_helio ** 2
    vinf2 = (-bb + math.sqrt(bb * bb - 4 * cc)) / 2
    dv = math.sqrt(vinf2 ** 2 + 2 * mu / rp) - math.sqrt(vinf ** 2 + 2 * mu / rp)
    return vinf2, dv


# --- 5. Задача Ламберта --------------------------------------------------------------

def _stumpff(z):
    if z > 1e-8:
        s = math.sqrt(z)
        return (1 - math.cos(s)) / z, (s - math.sin(s)) / s ** 3
    if z < -1e-8:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1) / (-z), (math.sinh(s) - s) / s ** 3
    return 0.5, 1 / 6


def lambert(r1, r2, dt, mu, prograde=True):
    R1, R2 = np.linalg.norm(r1), np.linalg.norm(r2)
    cross = r1[0] * r2[1] - r1[1] * r2[0]
    dth = math.acos(np.clip(np.dot(r1, r2) / (R1 * R2), -1, 1))
    if abs(dth) < 1e-6 or abs(dth - math.pi) < 1e-6:
        raise ValueError("вырожденная геометрия Ламберта")
    if prograde and cross < 0:
        dth = 2 * math.pi - dth
    A = math.sin(dth) * math.sqrt(R1 * R2 / (1 - math.cos(dth)))

    def t_of(z):
        C, S = _stumpff(z)
        y = R1 + R2 + A * (z * S - 1) / math.sqrt(C)
        if y < 0:
            return None
        x = math.sqrt(y / C)
        return (x ** 3 * S + A * math.sqrt(y)) / math.sqrt(mu)

    lo, hi = -4 * math.pi ** 2, 4 * math.pi ** 2
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        t = t_of(mid)
        if t is None or t < dt:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    C, S = _stumpff(z)
    y = R1 + R2 + A * (z * S - 1) / math.sqrt(C)
    f, g, gdot = 1 - y / R1, A * math.sqrt(y / mu), 1 - y / R2
    return (r2 - f * r1) / g, (gdot * r2 - r1) / g


def eeloo_pos(nu_deg, omega_deg=260.0):
    nu = math.radians(nu_deg)
    p = A_EELOO * (1 - E_EELOO ** 2)
    r = p / (1 + E_EELOO * math.cos(nu))
    x, y = r * math.cos(nu), r * math.sin(nu)
    c, s = math.cos(math.radians(omega_deg)), math.sin(math.radians(omega_deg))
    return np.array([c * x - s * y, s * x + c * y])


# --- 6. Графики ----------------------------------------------------------------------

def circle(r, n=600):
    th = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(th), r * np.sin(th)


def ellipse(a, e, omega=0.0, n=1200):
    th = np.linspace(0, 2 * np.pi, n)
    b = a * math.sqrt(max(1 - e * e, 1e-12))
    x, y = a * np.cos(th) - a * e, b * np.sin(th)
    c, s = math.cos(omega), math.sin(omega)
    return c * x - s * y, s * x + c * y


def conic_branch(r0, v0, mu=MU_JOOL, n=4000, r_max=3e9):
    """
    Аналитическая коника, построенная по начальному состоянию (r0, v0):
    через вектор эксцентриситета — так ветка автоматически совпадает
    с численной траекторией (без подгонки углов).
    """
    r0 = np.asarray(r0, float); v0 = np.asarray(v0, float)
    rn = np.linalg.norm(r0)
    h = r0[0] * v0[1] - r0[1] * v0[0]
    e_vec = ((v0 @ v0 - mu / rn) * r0 - (r0 @ v0) * v0) / mu
    e = np.linalg.norm(e_vec)
    p = h * h / mu
    phi_e = math.atan2(e_vec[1], e_vec[0])
    nu_max = math.acos(-1 / e) * 0.999 if e > 1 else math.pi
    nu = np.linspace(-nu_max, nu_max, n)
    r = p / (1 + e * np.cos(nu))
    ok = r < r_max
    return r[ok] * np.cos(phi_e + nu[ok]), r[ok] * np.sin(phi_e + nu[ok])


def fig_trajectory(tr, rp_2, rp_1):
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_aspect("equal")
    for r, c, l, ls in ((A_KERBIN, "#8ab4f8", "орбита Кербина", "-"),
                        (A_JOOL, "#34a853", "орбита Юла", "-")):
        x, y = circle(r)
        ax.plot(x / 1e9, y / 1e9, color=c, lw=1.2, ls=ls, label=l)
    x, y = ellipse(A_EELOO, E_EELOO, math.radians(260))
    ax.plot(x / 1e9, y / 1e9, color="#9aa0a6", lw=1.1, ls="--", label="орбита Эилу")
    x, y = ellipse(tr.a_t, (A_JOOL - A_KERBIN) / (A_JOOL + A_KERBIN))
    ax.plot(x / 1e9, y / 1e9, color="#ea4335", lw=2.2, label="Кербин → Юл")

    for rp, color, lab in ((rp_1, "#e37400", "после пролёта: уход из системы"),
                           (rp_2, "#9334e6", "после пролёта: цель — Эилу")):
        _, v_out = post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)
        el = helio_elements(np.array([A_JOOL, 0.0]), v_out)
        if math.isfinite(el["r_apo"]):
            x, y = ellipse(el["a"], el["e"], n=1500)
            ax.plot(x / 1e9, y / 1e9, color=color, lw=2, label=lab)
        else:
            # параболическая/гиперболическая ветка: строим как конику по r(nu)
            e, p = el["e"], A_JOOL * (1 + el["e"])
            nu_inf = math.acos(-1 / e) if e > 1 else math.pi
            nu = np.linspace(-min(nu_inf * 0.9, 2.6), min(nu_inf * 0.9, 2.6), 1500)
            r = p / (1 + e * np.cos(nu))
            ax.plot(r * np.cos(nu + math.pi) / 1e9, r * np.sin(nu + math.pi) / 1e9,
                    color=color, lw=2, label=lab)

    ax.plot(0, 0, "o", ms=14, color="#fbbc04", label="Кербол")
    ax.plot(A_JOOL / 1e9, 0, "o", ms=9, color="#34a853")
    ax.plot(A_KERBIN / 1e9, 0, "o", ms=7, color="#4285f4")
    ax.set_xlim(-1.2e11 / 1e9, 1.7e11 / 1e9)
    ax.set_ylim(-1.35e11 / 1e9, 1.35e11 / 1e9)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Схема миссии (2D): перелёт Кербин → Юл,\nпролёт Юла и послепролётные орбиты")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(os.path.join(OUT, "fig1_trajectory.png"))
    plt.close(fig)


def fig_vinf_map(tr, hs=np.linspace(200, 3e5, 900)):
    """Диаграмма проектирования: результат пролёта и необходимый импульс."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    apo, esc_h = [], []
    for h in hs:
        rp = R_JOOL + h * 1000
        _, v_out = post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)
        apo.append(helio_elements(np.array([A_JOOL, 0.0]), v_out)["r_apo"])
    apo = np.array([a if math.isfinite(a) else np.nan for a in apo])
    ax.plot(hs, apo / 1e9, color="#1a73e8", lw=2)
    ax.fill_between(hs, 0, np.nan_to_num(apo / 1e9), color="#1a73e8", alpha=0.10)
    ax.axhline(A_EELOO / 1e9, color="#9334e6", ls="--", lw=1.5,
               label="перигелий Эилу")
    ax.axhline(A_EELOO * (1 + E_EELOO) / 1e9, color="#34a853", ls="--", lw=1.5,
               label="афелий Эилу")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(150, 3.2e5)
    ax.set_xticks([200, 1000, 5000, 20000, 100000, 300000])
    ax.set_xticklabels(["200", "1k", "5k", "20k", "100k", "300k"])
    ax.set_xlabel("высота перицентра пролёта, км")
    ax.set_ylabel("афелий орбиты после пролёта, млн км")
    ax.set_title("Куда пролёт забрасывает аппарат:\nчем ближе к планете, тем сильнее поворот и дальше афелий")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    need_esc = []
    for h in hs:
        rp = R_JOOL + h * 1000
        d = delta_hyp(rp, tr.v_inf_j)
        v_esc_helio = v_esc(A_JOOL, MU_SUN)
        bb, cc = -2 * tr.v_jool * math.cos(d), tr.v_jool ** 2 - v_esc_helio ** 2
        disc = bb * bb - 4 * cc
        if disc >= 0:
            vinf_needed = (-bb + math.sqrt(disc)) / 2
            need_esc.append(math.sqrt(max(vinf_needed ** 2, 0) + 2 * MU_JOOL / rp)
                            - math.sqrt(tr.v_inf_j ** 2 + 2 * MU_JOOL / rp))
        else:
            need_esc.append(np.nan)
    need_esc = np.array(need_esc)
    ax.plot(hs, need_esc, color="#e37400", lw=2)
    ax.fill_between(hs, 0, np.nan_to_num(need_esc, nan=0), color="#e37400", alpha=0.12)
    ax.axvspan(0, ATM_JOOL / 1000, color="#fbbc04", alpha=0.35)
    ax.text(ATM_JOOL / 1000 * 1.15, np.nanmax(need_esc) * 0.55, "атмосфера\nЮла",
            fontsize=8, color="#b06000")
    ax.set_xscale("log")
    ax.set_xlim(150, 3.2e5)
    ax.set_xticks([200, 1000, 5000, 20000, 100000, 300000])
    ax.set_xticklabels(["200", "1k", "5k", "20k", "100k", "300k"])
    ax.set_xlabel("высота перицентра пролёта, км")
    ax.set_ylabel("Δv до ухода из системы, м/с")
    ax.set_title("Цена «билета» из системы Кербола:\nчем глубже пролёт, тем меньше нужный Δv")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_design_map.png"))
    plt.close(fig)


def fig_verify(tr):
    rp = R_JOOL + 20_000_000   # далёкий пролёт: кривизна видна целиком
    traj, vels, r_min, d_num, r_chk = propagate_hyperbola(rp, tr.v_inf_j, side=+1)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.plot(traj[:, 0], traj[:, 1], lw=1.6, color="#1a73e8", label="RK4")
    b = b_hyp(rp, tr.v_inf_j)
    r0 = np.array([-math.sqrt(r_chk ** 2 - b ** 2), b])
    xh, yh = conic_branch(r0, np.array([tr.v_inf_j, 0.0]), r_max=r_chk)
    ax.plot(xh, yh, "--", lw=1.2, color="#ea4335", label="аналитика (коника)")
    ax.plot(0, 0, "o", ms=16, color="#34a853", label="Юл")
    ax.set_aspect("equal", adjustable="box")
    bb = b_hyp(rp, tr.v_inf_j)
    ax.set_xlim(-8 * bb, 2 * bb); ax.set_ylim(-9 * bb, 3 * bb)
    ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
    ax.set_xlabel("x, км"); ax.set_ylabel("y, км")
    ax.set_title(f"Пролёт Юла (h_p = 20 000 км):\nчисленно (RK4, адаптивный шаг) и аналитически")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    rr = np.linspace(R_JOOL + ATM_JOOL, 40 * R_JOOL, 400)
    ax.plot((rr - R_JOOL) / 1000, [math.degrees(delta_hyp(r, tr.v_inf_j)) for r in rr],
            color="#1a73e8", lw=1.8, label="δ = 2·arcsin(1/e)")
    ax.plot((rp - R_JOOL) / 1000, math.degrees(d_num), "o", color="#ea4335", ms=8,
            label=f"RK4 + конечный радиус: {math.degrees(d_num):.2f}°")
    ax.plot((rp - R_JOOL) / 1000, math.degrees(delta_hyp(rp, tr.v_inf_j)), "x",
            color="#34a853", ms=10, mew=2,
            label=f"асимптотика 2·arcsin(1/e): {math.degrees(delta_hyp(rp, tr.v_inf_j)):.2f}°")
    ax.set_xscale("log")
    ax.set_xlabel("высота перицентра, км"); ax.set_ylabel("δ, °")
    err = abs(math.degrees(d_num - delta_hyp(rp, tr.v_inf_j)))
    ax.set_title(f"Верификация модели пролёта: численно vs аналитика\n"
                 f"δ(конечн. радиус) = {math.degrees(d_num):.2f}°, "
                 f"r_p(числ.) = {r_min/1e3:.1f} км (задано {rp/1e3:.0f} км)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_verification.png"))
    plt.close(fig)
    return r_min, d_num, err


def fig_oberth(tr, rp=200_000):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    dlt = np.linspace(0, 180, 600)
    v_out = np.sqrt(tr.v_jool ** 2 + tr.v_inf_j ** 2
                    - 2 * tr.v_jool * tr.v_inf_j * np.cos(np.radians(dlt)))
    ax.plot(dlt, v_out, color="#1a73e8", lw=2, label="|v| после пролёта")
    ax.axhline(v_esc(A_JOOL, MU_SUN), color="#9334e6", ls="--", lw=1.6,
               label="скорость выхода из системы")
    ax.axhline(tr.v_jool, color="#9aa0a6", ls=":", lw=1.2, label="скорость Юла (δ = 0)")
    for h in (200, 5000, 50_000):
        d = math.degrees(delta_hyp(R_JOOL + h * 1e3, tr.v_inf_j))
        vv = np.linalg.norm(post_flyby(R_JOOL + h * 1e3, tr.v_inf_j, +1, tr.v_jool)[1])
        ax.plot(d, vv, "o", ms=8, label=f"h_p = {h:,} км".replace(",", " "))
    ax.set_xlabel("угол поворота δ, °")
    ax.set_ylabel("гелиоцентрическая скорость после пролёта, м/с")
    ax.set_title("«Треугольник скоростей» гравитационного манёвра:\n|v∞| неизменна, поворачивается только её направление")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    hs2 = np.linspace(200, 40000, 300)
    dvs = []
    for h in hs2:
        d = delta_hyp(R_JOOL + h * 1e3, tr.v_inf_j)
        _, dv = oberth_dv(R_JOOL + h * 1e3, tr.v_inf_j, d, tr.v_jool)
        dvs.append(dv)
    ax.plot(hs2, dvs, color="#1a73e8", lw=2)
    ax.fill_between(hs2, dvs, color="#1a73e8", alpha=0.12)
    ax.axvline(ATM_JOOL / 1000, color="#fbbc04", lw=4, alpha=0.5)
    ax.text(ATM_JOOL / 1000 * 1.15, 1200, "атмосфера", fontsize=8, color="#b06000")
    ax.set_xlabel("высота перицентра, км")
    ax.set_ylabel("Δv до выхода из системы, м/с")
    ax.set_title("Эффект Оберта: недостающий поворот «добирается»\nимпульсом в перицентре — чем глубже, тем дешевле")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_oberth.png"))
    plt.close(fig)


def fig_lambert(tr):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    pts = []
    for nu in range(0, 360, 5):
        r2 = eeloo_pos(nu)
        for tof_days in (400, 700, 1000):
            try:
                v1, _ = lambert(np.array([A_JOOL, 0.0]), r2, tof_days * DAY, MU_SUN)
            except Exception:
                continue
            vinf = np.linalg.norm(v1 - np.array([0.0, tr.v_jool]))
            v_esc_helio = v_esc(np.linalg.norm(np.array([A_JOOL, 0.0])), MU_SUN)
            if np.linalg.norm(v1) > v_esc_helio:
                continue  # такая траектория не реализуема (уход из системы)
            pts.append((tof_days, np.linalg.norm(r2) / 1e9, vinf, nu))
    p = np.array(pts)
    for tof_days, c in ((400, "#34a853"), (700, "#1a73e8"), (1000, "#9334e6")):
        m = p[:, 0] == tof_days
        o = np.argsort(p[m][:, 3])
        ax.plot(p[m][o][:, 1], p[m][o][:, 2], "o-", ms=2.5, lw=1.4, color=c,
                label=f"перелёт {tof_days} сут")
    ax.axhline(tr.v_inf_j, color="#ea4335", ls="--", lw=1.7,
               label=f"располагаемая v∞ = {tr.v_inf_j:.0f} м/с")
    ax.axvline(A_EELOO / 1e9, color="#9aa0a6", ls=":", lw=1)
    ax.set_ylim(0, 4000)
    ax.set_xlabel("расстояние Юл — Эилу в точке встречи, млн км")
    ax.set_ylabel("требуемая v∞, м/с")
    ax.set_title("Задача Ламберта (Юл → Эилу): какие стартовые условия попадают\n"
                 "в «коридор» реально располагаемой v∞")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT, "fig5_lambert.png"))
    plt.close(fig)
    return p



# --- 6a. Проверка на реальных данных (Юпитер / «Вояджер-2», NASA JPL Horizons) --------

JUP = dict(mu=1.26686534e8, R=71_492.0,          # км, км/с (Юпитер, IAU)
           r_p=721_895.0,                        # перицентр по 2-часовым эфемеридам
           v_inf=7.60,                           # из фита vis-viva (кольцо 20-120 R_J)
           v_before=9.863, v_after=20.029,       # гелиоцентрические скорости (JPL)
           delta_obs=91.16,                      # измеренный угол на r ≈ 60 R_J
           r_obs=60.0)


def delta_at_radius(rp, vinf, rr, mu=MU_JOOL):
    """Угол между векторами скорости на входе и выходе на ОДНОМ радиусе rr."""
    e = 1 + rp * vinf ** 2 / mu
    p = rp * (1 + e)
    cosnu = (p / rr - 1) / e
    nu = math.acos(max(-1.0, min(1.0, cosnu)))
    s, c = math.sin(nu), math.cos(nu)
    num = (e + c) ** 2 - s ** 2
    den = (e + c) ** 2 + s ** 2
    return math.degrees(math.acos(max(-1.0, min(1.0, num / den))))


def check_real_world():
    """
    Проверка модели на реальных данных NASA JPL Horizons (пролёт «Вояджера-2»
    у Юпитера, 09.07.1979): v∞ из закона сохранения энергии, e, δ.
    Модель воспроизводит измеренный на r = 60 R_J угол поворота.
    """
    mu, rp, vinf = JUP["mu"], JUP["r_p"], JUP["v_inf"]
    e = 1 + rp * vinf ** 2 / mu
    delta_as = 2 * math.asin(1 / e)
    delta_obs_model = delta_at_radius(rp, vinf, JUP["r_obs"] * JUP["R"], mu)
    return dict(e=e, delta_asymptotic=math.degrees(delta_as),
                delta_model_at_r=delta_obs_model, delta_observed=JUP["delta_obs"],
                dv_actual=JUP["v_after"] - JUP["v_before"],
                rp_over_R=rp / JUP["R"],
                b_km=rp * math.sqrt((e + 1) / (e - 1)))


def fig_scaling(tr, real):
    """Сопоставление «игрового» и реального пролёта."""
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax = axes[0]
    rr = np.linspace(1.05, 12, 500)
    ax.plot(rr, [math.degrees(delta_hyp(r * R_JOOL, tr.v_inf_j)) for r in rr],
            lw=2, color="#1a73e8", label="Юл (Кербол), v∞ = 1756 м/с")
    ax.plot(rr, [math.degrees(delta_hyp(r * JUP["R"], JUP["v_inf"], JUP["mu"]))
                 for r in rr], lw=2, color="#ea4335",
            label="Юпитер, v∞ = 7.6 км/с («Вояджер-2»)")
    ax.plot(real["rp_over_R"], real["delta_asymptotic"], "*", ms=17, color="#ea4335")
    ax.plot(real["rp_over_R"], JUP["delta_obs"], "o", ms=8, mfc="white", mec="#ea4335", mew=2)
    ax.annotate("«Вояджер-2»: 10.1 R$_J$\nмодель 97.6°, факт 91.2°",
                (real["rp_over_R"], real["delta_observed"]),
                xytext=(-150, 30), textcoords="offset points", fontsize=8, color="#b31412",
                arrowprops=dict(arrowstyle="->", color="#b31412", lw=1))
    ax.set_xlabel("перицентр, r$_p$/R")
    ax.set_ylabel("угол поворота δ, °")
    ax.set_ylim(60, 185)
    ax.set_title("Универсальная кривая δ = 2·arcsin(1/e):\nодна формула для игры и для реального полёта")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    names = ["Юл:\nдо", "Юл:\nпосле", "Юпитер:\nдо", "Юпитер:\nпосле"]
    vals = [tr.v_jool,
            np.linalg.norm(post_flyby(R_JOOL + 300e3, tr.v_inf_j, +1, tr.v_jool)[1]),
            JUP["v_before"] * 1e3, JUP["v_after"] * 1e3]
    cols = ["#1a73e8", "#1a73e8", "#ea4335", "#ea4335"]
    bars = ax.bar(names, np.array(vals) / 1000, color=cols, alpha=0.85, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v / 1000 + 0.3, f"{v/1000:.2f}",
                ha="center", fontsize=9)
    ax.set_ylabel("гелиоцентрическая скорость, км/с")
    ax.set_ylim(0, 24)
    ax.set_title("Энергия от пролёта: модель (h_p = 300 км) и\nреальный пролёт «Вояджера-2» (09.07.1979)")
    ax.text(0.02, 0.95, f"модель: +{(vals[1]-vals[0])/1000:.2f} км/с\n"
                        f"реальность: +{(vals[3]-vals[2])/1000:.2f} км/с",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f1f3f4", ec="#dadce0"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_scaling.png"))
    plt.close(fig)


# --- 7. Основной расчёт --------------------------------------------------------------

def main():
    tr = launch_window()
    print("=" * 80)
    print("1. ПЕРЕЛЁТ КЕРБИН → ЮЛ")
    print("=" * 80)
    print(f"  a перелёта             {tr.a_t/1e9:9.3f} млн км")
    print(f"  время перелёта         {tr.tof/DAY:9.1f} сут = {tr.tof/KSP_DAY:.0f} керб. сут")
    print(f"  v∞ отлёта от Кербина   {tr.v_inf_k:9.1f} м/с")
    print(f"  Δv отлёта с орбиты 100 км {tr.dv_tmi:6.1f} м/с")
    print(f"  скорость Юла           {tr.v_jool:9.1f} м/с")
    print(f"  v∞ прилёта к Юлу       {tr.v_inf_j:9.1f} м/с")
    print(f"  угол опережения Юла    {tr.phase_deg:9.2f}°")

    print("\n" + "=" * 80)
    print("2. ПРОЛЁТ ЮЛА")
    print("=" * 80)
    print(f"  {'h_p, км':>9} {'δ, °':>8} {'b, тыс.км':>11} {'афелий, млн км':>15} {'e':>7}")
    for h in (200, 500, 1_000, 5_000, 20_000, 100_000, 500_000):
        rp = R_JOOL + h * 1e3
        el = helio_elements(np.array([A_JOOL, 0.0]), post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)[1])
        print(f"  {h:9d} {math.degrees(delta_hyp(rp, tr.v_inf_j)):8.2f} "
              f"{b_hyp(rp, tr.v_inf_j)/1e3:11.0f} {el['r_apo']/1e9:15.2f} {el['e']:7.3f}")

    rp_target = rp_for_apo(A_EELOO * (1 + E_EELOO), tr.v_inf_j, +1, tr.v_jool)
    print(f"\n  Для афелия = афелий Эилу ({A_EELOO*(1+E_EELOO)/1e9:.1f} млн км):")
    if rp_target:
        el = helio_elements(np.array([A_JOOL, 0.0]), post_flyby(rp_target, tr.v_inf_j, +1, tr.v_jool)[1])
        print(f"    h_p = {(rp_target-R_JOOL)/1e3:.0f} км, δ = {math.degrees(delta_hyp(rp_target, tr.v_inf_j)):.2f}°, "
              f"a = {el['a']/1e9:.2f} млн км, e = {el['e']:.3f}")
        print(f"    (выше атмосферы Юла: {'да' if rp_target > R_JOOL + ATM_JOOL else 'НЕТ'})")

    v_esc_helio = v_esc(A_JOOL, MU_SUN)
    print(f"\n  Уход из системы Кербола: нужно |v| ≥ {v_esc_helio:.0f} м/с на r = r_Юла")
    need_delta = math.degrees(math.acos((tr.v_jool**2 + tr.v_inf_j**2 - v_esc_helio**2)
                                        / (2 * tr.v_jool * tr.v_inf_j)))
    print(f"  требуемый угол поворота δ ≥ {need_delta:.2f}°")
    print(f"  достижимый максимум δ = {math.degrees(delta_hyp(R_JOOL + ATM_JOOL, tr.v_inf_j)):.2f}° "
          f"→ только пролётом систему не покинуть")
    vinf2, dv_ob = oberth_dv(R_JOOL + 200e3, tr.v_inf_j,
                             delta_hyp(R_JOOL + 200e3, tr.v_inf_j), tr.v_jool)
    print(f"  Решение: доп. импульс в перицентре 200 км: v∞ {tr.v_inf_j:.0f} → {vinf2:.0f} м/с, "
          f"Δv = {dv_ob:.1f} м/с")

    r_min, d_num, err_num = fig_verify(tr)
    print(f"\n  Верификация RK4 (пролёт 20 000 км, старт на 0.95 SOI Юла):")
    print(f"    δ численно (на конечном радиусе) = {math.degrees(d_num):.3f}°")
    print(f"    δ асимптотика 2·arcsin(1/e)      = {math.degrees(delta_hyp(R_JOOL + 20e6, tr.v_inf_j)):.3f}°")
    print(f"    расхождение                      = {err_num:.3f}°;  "
          f"r_p(числ) = {r_min/1e3:.0f} км (задано 26000)")

    rp_1 = R_JOOL + 200e3
    fig_trajectory(tr, rp_target or R_JOOL + 1.5e6, rp_1)
    fig_vinf_map(tr)
    fig_oberth(tr)
    pts = fig_lambert(tr)

    m_ok = pts[:, 2] <= tr.v_inf_j
    print("\n" + "=" * 80)
    print("3. ЗАДАЧА ЛАМБЕРТА (Юл → Эилу)")
    print("=" * 80)
    print(f"  проанализировано {len(pts)} сочетаний (фаза Эилу × время перелёта)")
    print(f"  попадают в коридор v∞ ≤ {tr.v_inf_j:.0f} м/с: {m_ok.sum()} "
          f"({100*m_ok.mean():.1f}%)")
    if m_ok.any():
        best = pts[m_ok]
        i = np.argmin(best[:, 2])
        print(f"  лучшее: TOF = {best[i,0]:.0f} сут, расстояние {best[i,1]:.1f} млн км, "
              f"требуемая v∞ = {best[i,2]:.0f} м/с")

    real = check_real_world()
    print("\n" + "=" * 80)
    print("4. ПРОВЕРКА НА РЕАЛЬНЫХ ДАННЫХ (Юпитер, «Вояджер-2», 09.07.1979)")
    print("=" * 80)
    print(f"  r_p = {JUP['r_p']:.0f} км = {real['rp_over_R']:.2f} R_Юпитера;  v∞ = {JUP['v_inf']:.2f} км/с")
    print(f"  e = 1 + r_p·v∞²/μ = {real['e']:.3f};  δ(асимптота) = 2·arcsin(1/e) = {real['delta_asymptotic']:.2f}°")
    print(f"  угол между векторами скорости на r = 60 R_Юпитера:")
    print(f"    модель  = {real['delta_model_at_r']:.2f}°")
    print(f"    по эфемеридам JPL = {real['delta_observed']:.2f}°  "
          f"(расхождение {abs(real['delta_model_at_r']-real['delta_observed']):.2f}°)")
    print(f"  прирост гелиоцентрической скорости (факт): "
          f"{JUP['v_before']:.2f} → {JUP['v_after']:.2f} км/с (+{real['dv_actual']:.2f})")
    fig_scaling(tr, real)

    summary = dict(transfer=asdict(tr),
                   real_world_check=real,
                   flyby=dict(h_p_target_km=(rp_target - R_JOOL) / 1e3 if rp_target else None,
                              delta_deg=math.degrees(delta_hyp(rp_target or 0, tr.v_inf_j)),
                              v_escape=v_esc_helio, need_delta_deg=need_delta,
                              oberth_dv=dv_ob,
                              rk4_check=dict(delta_num_deg=math.degrees(d_num),
                                             delta_asymptotic_deg=math.degrees(
                                                 delta_hyp(R_JOOL + 20e6, tr.v_inf_j)),
                                             err_deg=err_num,
                                             rp_num_km=r_min / 1e3)))
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nГрафики сохранены: {OUT}")


if __name__ == "__main__":
    main()
