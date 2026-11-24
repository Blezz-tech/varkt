"""
Физико-математическая модель гравитационного манёвра у газового гиганта
(пролёт Юла в системе Кербола и верификация по пролёту «Вояджера-2» у Юпитера).

Модель: плоская (2D), метод точечных сфер действия (patched conics).
Участки:
  1) гомановский перелёт Кербин -> Юл в поле Кербола;
  2) гиперболический пролёт в сфере действия Юла (аналитика + численный метод RK4);
  3) послепролётная гелиоцентрическая орбита («треугольник скоростей»);
  4) условие ухода из системы и импульс в перицентре (эффект Оберта);
  5) верификация по реальным эфемеридам NASA JPL Horizons («Вояджер-2», Юпитер).

Единицы: СИ (для реального Юпитера в блоке JPL — км и км/с).
Запуск:  python3 src/sim.py
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

# --- 1. Физические константы системы Кербола (KSP 1.12, СИ) --------------------------
MU_SUN = 1.1723328e18
MU_KERBIN, R_KERBIN, A_KERBIN = 3.5316e12, 600_000.0, 13_599_840_256.0
MU_JOOL, R_JOOL, A_JOOL = 2.82528e14, 6_000_000.0, 68_773_560_320.0
SOI_JOOL, ATM_JOOL = 2_455_985_200.0, 200_000.0
A_EELOO, E_EELOO = 90_118_820_000.0, 0.26
DAY, KSP_DAY = 86_400.0, 21_600.0


def period(a: float, mu: float) -> float:
    return 2 * math.pi * math.sqrt(a ** 3 / mu)


def v_circ(r: float, mu: float) -> float:
    return math.sqrt(mu / r)


def v_esc(r: float, mu: float) -> float:
    return math.sqrt(2 * mu / r)


# --- 2. Гомановский перелёт Кербин -> Юл ---------------------------------------------

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


def launch_window(r1: float = R_KERBIN + 100_000.0) -> Transfer:
    """Расчёт полуэллипса Гомана от орбиты Кербина до орбиты Юла."""
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


# --- 3. Гиперболический пролёт в сфере действия Юла ----------------------------------

def e_hyp(rp: float, vinf: float, mu: float = MU_JOOL) -> float:
    """Эксцентриситет пролётной гиперболы: e = 1 + r_p * v_inf^2 / mu."""
    return 1.0 + rp * vinf ** 2 / mu


def delta_hyp(rp: float, vinf: float, mu: float = MU_JOOL) -> float:
    """Асимптотический угол поворота вектора скорости: delta = 2 * arcsin(1/e)."""
    return 2 * math.asin(1.0 / e_hyp(rp, vinf, mu))


def b_hyp(rp: float, vinf: float, mu: float = MU_JOOL) -> float:
    """Прицельный параметр: b = r_p * sqrt((e + 1) / (e - 1))."""
    e = e_hyp(rp, vinf, mu)
    return rp * math.sqrt((e + 1) / (e - 1))


def delta_at_radius(rp: float, vinf: float, rr: float, mu: float = MU_JOOL) -> float:
    """Угол (в градусах) между векторами скорости на входе и выходе на конечном радиусе rr."""
    e = e_hyp(rp, vinf, mu)
    p = rp * (1 + e)
    cosnu = (p / rr - 1) / e
    nu = math.acos(max(-1.0, min(1.0, cosnu)))
    s, c = math.sin(nu), math.cos(nu)
    num = (e + c) ** 2 - s ** 2
    den = (e + c) ** 2 + s ** 2
    return math.degrees(math.acos(max(-1.0, min(1.0, num / den))))


def state_on_hyperbola(rp: float, vinf: float, rr: float, branch: int,
                       mu: float = MU_JOOL) -> tuple[np.ndarray, np.ndarray]:
    """Точное положение и скорость на гиперболе в точке радиуса rr (branch=-1 вход, +1 выход)."""
    e = e_hyp(rp, vinf, mu)
    p = rp * (1 + e)
    cosnu = (p / rr - 1) / e
    nu = branch * math.acos(max(-1.0, min(1.0, cosnu)))
    er = np.array([math.cos(nu), math.sin(nu)])
    et = np.array([-math.sin(nu), math.cos(nu)])
    gamma = math.atan2(e * math.sin(nu), 1 + e * math.cos(nu))
    speed = math.sqrt(vinf ** 2 + 2 * mu / rr)
    return rr * er, speed * (math.sin(gamma) * er + math.cos(gamma) * et)


def propagate_hyperbola(rp: float, vinf: float, mu: float = MU_JOOL,
                        frac_soi: float = 0.95, dt_max: float = 120.0,
                        soi: float = SOI_JOOL):
    """
    Численное интегрирование задачи двух тел методом Рунге—Кутты 4-го порядка (RK4)
    с адаптивным шагом внутри сферы действия Юла (старт и финиш на r = frac_soi * SOI).
    Возвращает траекторию, численный перицентр r_min и численный угол поворота вектора скорости.
    """
    r_check = frac_soi * soi
    r, v = state_on_hyperbola(rp, vinf, r_check, branch=-1, mu=mu)
    v_start = v.copy()

    def acc(q):
        return -mu * q / np.dot(q, q) ** 1.5

    traj = [r.copy()]
    r_min, n = float(np.linalg.norm(r)), 0
    while n < 2_000_000:
        rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
        dt = min(dt_max, max(0.5, 0.01 * rn / vn))
        a1 = acc(r);               k1 = v
        a2 = acc(r + 0.5 * dt * k1); k2 = v + 0.5 * dt * a1
        a3 = acc(r + 0.5 * dt * k2); k3 = v + 0.5 * dt * a2
        a4 = acc(r + dt * k3);       k4 = v + dt * a3
        r = r + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        v = v + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        rn_new = float(np.linalg.norm(r))
        r_min = min(r_min, rn_new)
        n += 1
        if np.linalg.norm(r - traj[-1]) > 0.004 * rn_new:
            traj.append(r.copy())
        if rn_new >= r_check and np.dot(r, v) > 0:
            break
    traj.append(r.copy())

    cos_d = np.clip(np.dot(v_start, v) / (np.linalg.norm(v_start) * np.linalg.norm(v)), -1.0, 1.0)
    delta_rk4 = math.acos(float(cos_d))
    return np.array(traj), r_min, delta_rk4, r_check


# --- 4. Послепролётная гелиоцентрическая орбита и эффект Оберта ----------------------

def turn_vinf(vinf_in: np.ndarray, delta: float, side: int) -> np.ndarray:
    """Поворот вектора v_inf на угол delta (side = +1 позади планеты — разгон)."""
    c, s = math.cos(side * delta), math.sin(side * delta)
    return np.array([[c, -s], [s, c]]) @ vinf_in


def post_flyby(rp: float, vinf: float, side: int, v_j: float):
    """
    Треугольник скоростей: до пролёта аппарат медленнее Юла, поэтому
    v_inf_in = (0, -vinf) направлена против скорости планеты v_J = (0, v_j).
    После поворота на угол delta получаем v_out = v_J + v_inf_out.
    """
    vinf_in = np.array([0.0, -vinf])
    vinf_out = turn_vinf(vinf_in, delta_hyp(rp, vinf), side)
    return vinf_out, np.array([0.0, v_j]) + vinf_out


def helio_elements(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_SUN) -> dict:
    """Кеплеровы элементы послепролётной гелиоцентрической орбиты по векторам r и v."""
    r, v = float(np.linalg.norm(r_vec)), float(np.linalg.norm(v_vec))
    eps = v * v / 2 - mu / r
    h = float(r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0])
    a = -mu / (2 * eps) if eps < 0 else math.inf
    e_vec = ((v * v - mu / r) * r_vec - float(np.dot(r_vec, v_vec)) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    r_per = h * h / mu / (1 + e)
    return dict(eps=eps, a=a, e=e, r_per=r_per,
                r_apo=a * (1 + e) if math.isfinite(a) else math.inf)


def rp_for_apo(apo_target: float, vinf: float, side: int, v_j: float,
               r_helio: float = A_JOOL) -> float | None:
    """Подбор радиуса перицентра пролёта под заданный афелий методом бисекции."""
    def f(rp):
        _, v_out = post_flyby(rp, vinf, side, v_j)
        return helio_elements(np.array([r_helio, 0.0]), v_out)["r_apo"] - apo_target

    lo, hi = R_JOOL + ATM_JOOL + 10_000, 200 * R_JOOL
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def oberth_dv(rp: float, vinf: float, delta: float, v_j: float,
              mu: float = MU_JOOL) -> tuple[float, float]:
    """Импульс Delta_v в перицентре гиперболы для достижения второй космической скорости."""
    v_esc_helio = v_esc(A_JOOL, MU_SUN)
    # Квадратное уравнение из теоремы косинусов: v_inf'^2 - 2 v_j cos(delta) v_inf' + v_j^2 - v_esc^2 = 0
    bb = -2 * v_j * math.cos(delta)
    cc = v_j ** 2 - v_esc_helio ** 2
    vinf2 = (-bb + math.sqrt(bb * bb - 4 * cc)) / 2
    dv = math.sqrt(vinf2 ** 2 + 2 * mu / rp) - math.sqrt(vinf ** 2 + 2 * mu / rp)
    return vinf2, dv


# --- 5. Верификация по реальным данным («Вояджер-2» у Юпитера, JPL Horizons) ---------

def load_voyager_reference() -> dict:
    """Читает параметры пролёта «Вояджера-2» из снимка JPL Horizons (или встроенные константы)."""
    snap = os.path.join(OUT, "jpl_snapshot.json")
    base = dict(mu=1.26686534e8, R=71_492.0,
                r_p=721_894.9, v_inf=7.6055,
                v_before=9.863, v_after=20.029,
                delta_obs=91.16, r_obs_R=60.15)
    if os.path.exists(snap):
        with open(snap, encoding="utf-8") as f:
            d = json.load(f)
        fb, hel = d["flyby"], d.get("heliocentric", {})
        base.update(
            r_p=fb["r_p_km"],
            v_inf=fb["v_inf_km_s"],
            delta_obs=fb["delta_observed"],
            r_obs_R=fb["r_obs_before_R"],
            v_before=hel.get("v_before_km_s", base["v_before"]),
            v_after=hel.get("v_after_km_s", base["v_after"]),
        )
    return base


def check_real_world() -> dict:
    """Проверка аналитической модели пролёта на эфемеридах «Вояджера-2» (09.07.1979)."""
    jup = load_voyager_reference()
    mu, rp, vinf = jup["mu"], jup["r_p"], jup["v_inf"]
    e = e_hyp(rp, vinf, mu)
    delta_as = math.degrees(delta_hyp(rp, vinf, mu))
    delta_mod = delta_at_radius(rp, vinf, jup["r_obs_R"] * jup["R"], mu)
    return dict(
        r_p_km=rp, rp_over_R=rp / jup["R"], v_inf_km_s=vinf,
        e=e, delta_asymptotic=delta_as,
        delta_model_at_r=delta_mod, delta_observed=jup["delta_obs"],
        err_deg=abs(delta_mod - jup["delta_obs"]),
        v_before=jup["v_before"], v_after=jup["v_after"],
        dv_actual=jup["v_after"] - jup["v_before"],
        b_km=b_hyp(rp, vinf, mu), R_km=jup["R"], mu=mu,
    )


# --- 6. Построение рисунков для отчёта -----------------------------------------------

def _circle(r: float, n: int = 600):
    th = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(th), r * np.sin(th)


def _ellipse(a: float, e: float, omega: float = 0.0, n: int = 1200):
    th = np.linspace(0, 2 * np.pi, n)
    b = a * math.sqrt(max(1 - e * e, 1e-12))
    x, y = a * np.cos(th) - a * e, b * np.sin(th)
    c, s = math.cos(omega), math.sin(omega)
    return c * x - s * y, s * x + c * y


def _conic_branch(r0: np.ndarray, v0: np.ndarray, mu: float = MU_JOOL,
                  n: int = 4000, r_max: float = 3e9):
    """Аналитическая коника (эллипс или гипербола) по вектору состояния (r0, v0)."""
    rn = float(np.linalg.norm(r0))
    h = float(r0[0] * v0[1] - r0[1] * v0[0])
    e_vec = ((float(v0 @ v0) - mu / rn) * r0 - float(r0 @ v0) * v0) / mu
    e = float(np.linalg.norm(e_vec))
    p = h * h / mu
    phi_e = math.atan2(float(e_vec[1]), float(e_vec[0]))
    if e >= 1.0:
        nu_max = math.acos(-1 / e) * 0.999
        nu = np.linspace(-nu_max, nu_max, n)
        r = p / (1 + e * np.cos(nu))
        ok = r < r_max
        return r[ok] * np.cos(phi_e + nu[ok]), r[ok] * np.sin(phi_e + nu[ok])
    nu = np.linspace(0, 2 * math.pi, n)
    r = p / (1 + e * np.cos(nu))
    return r * np.cos(phi_e + nu), r * np.sin(phi_e + nu)


def fig_trajectory(tr: Transfer, rp_eeloo: float, rp_low: float):
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_aspect("equal")
    for r, c, lab in ((A_KERBIN, "#8ab4f8", "орбита Кербина"),
                      (A_JOOL, "#34a853", "орбита Юла")):
        x, y = _circle(r)
        ax.plot(x / 1e9, y / 1e9, color=c, lw=1.2, label=lab)
    x, y = _ellipse(A_EELOO, E_EELOO, math.radians(260))
    ax.plot(x / 1e9, y / 1e9, color="#9aa0a6", lw=1.1, ls="--", label="орбита Эилу")

    # Встреча с Юлом происходит в точке (-A_JOOL, 0), где скорость Юла v_J = (0, -v_jool)
    # Старт от Кербина в точке (+A_KERBIN, 0) в направлении +y -> полуэллипс сверху до (-A_JOOL, 0)
    th_tr = np.linspace(0, math.pi, 800)
    e_tr = (A_JOOL - A_KERBIN) / (A_JOOL + A_KERBIN)
    r_tr = tr.a_t * (1 - e_tr ** 2) / (1 + e_tr * np.cos(th_tr))
    ax.plot(r_tr * np.cos(th_tr) / 1e9, r_tr * np.sin(th_tr) / 1e9,
            color="#ea4335", lw=2.3, label="гомановский перелёт Кербин → Юл")

    r_meet = np.array([-A_JOOL, 0.0])
    for rp, color, lab in ((rp_low, "#e37400", "после пролёта h_p = 200 км (афелий 711 млн км)"),
                           (rp_eeloo, "#9334e6", "после пролёта h_p = 35 096 км (к афелию Эилу)")):
        # В точке (-A_JOOL, 0) Юл движется в сторону -y; поворачиваем результат post_flyby на 180°
        _, v_out_std = post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)
        v_out = -v_out_std
        # Рисуем ветвь от момента пролёта Юла до афелия и обратно
        xh, yh = _conic_branch(r_meet, v_out, mu=MU_SUN, n=2500)
        ax.plot(xh / 1e9, yh / 1e9, color=color, lw=1.9, label=lab)

    ax.plot(0, 0, "o", ms=14, color="#fbbc04", label="Кербол")
    ax.plot(-A_JOOL / 1e9, 0, "o", ms=9, color="#34a853", label="точка встречи с Юлом")
    ax.plot(A_KERBIN / 1e9, 0, "o", ms=7, color="#4285f4", label="старт с Кербина")
    ax.set_xlim(-125, 165)
    ax.set_ylim(-135, 135)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Схема миссии (2D): перелёт Кербин → Юл,\nгравитационный пролёт и послепролётные орбиты")
    ax.legend(loc="upper right", fontsize=7.6)
    fig.savefig(os.path.join(OUT, "fig1_trajectory.png"))
    plt.close(fig)


def fig_design_map(tr: Transfer, hs=np.linspace(200, 3e5, 900)):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    apo = []
    for h in hs:
        rp = R_JOOL + h * 1000
        _, v_out = post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)
        apo.append(helio_elements(np.array([A_JOOL, 0.0]), v_out)["r_apo"])
    apo = np.array(apo)
    ax.plot(hs, apo / 1e9, color="#1a73e8", lw=2)
    ax.fill_between(hs, 0, apo / 1e9, color="#1a73e8", alpha=0.10)
    ax.axhline(A_EELOO / 1e9, color="#9334e6", ls="--", lw=1.5, label="большая полуось Эилу")
    ax.axhline(A_EELOO * (1 + E_EELOO) / 1e9, color="#34a853", ls="--", lw=1.5, label="афелий Эилу (113.5 млн км)")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(150, 3.2e5)
    ax.set_xticks([200, 1000, 5000, 20000, 100000, 300000])
    ax.set_xticklabels(["200", "1k", "5k", "20k", "100k", "300k"])
    ax.set_xlabel("высота перицентра пролёта h_p, км")
    ax.set_ylabel("афелий орбиты после пролёта, млн км")
    ax.set_title("Эффективность гравитационного манёвра:\nчем ближе пролёт, тем сильнее поворот и выше афелий")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    need_esc = []
    for h in hs:
        rp = R_JOOL + h * 1000
        _, dv = oberth_dv(rp, tr.v_inf_j, delta_hyp(rp, tr.v_inf_j), tr.v_jool)
        need_esc.append(dv)
    need_esc = np.array(need_esc)
    ax.plot(hs, need_esc, color="#e37400", lw=2)
    ax.fill_between(hs, 0, need_esc, color="#e37400", alpha=0.12)
    ax.axvspan(0, ATM_JOOL / 1000, color="#fbbc04", alpha=0.35)
    ax.text(ATM_JOOL / 1000 * 1.15, float(np.max(need_esc)) * 0.55, "атмосфера\nЮла",
            fontsize=8, color="#b06000")
    ax.set_xscale("log")
    ax.set_xlim(150, 3.2e5)
    ax.set_xticks([200, 1000, 5000, 20000, 100000, 300000])
    ax.set_xticklabels(["200", "1k", "5k", "20k", "100k", "300k"])
    ax.set_xlabel("высота перицентра пролёта h_p, км")
    ax.set_ylabel("импульс Δv в перицентре для ухода из системы, м/с")
    ax.set_title("Цена ухода из системы Кербола:\nчем глубже перицентр, тем меньше нужный Δv")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_design_map.png"))
    plt.close(fig)


def fig_verify(tr: Transfer):
    rp = R_JOOL + 20_000_000.0
    traj, r_min, d_rk4, r_chk = propagate_hyperbola(rp, tr.v_inf_j)
    d_fin_an = delta_at_radius(rp, tr.v_inf_j, r_chk)
    d_asym = math.degrees(delta_hyp(rp, tr.v_inf_j))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.plot(traj[:, 0] / 1e3, traj[:, 1] / 1e3, lw=1.8, color="#1a73e8", label="численно (RK4)")
    r0, v0 = state_on_hyperbola(rp, tr.v_inf_j, r_chk, branch=-1)
    xh, yh = _conic_branch(r0, v0, r_max=r_chk)
    ax.plot(xh / 1e3, yh / 1e3, "--", lw=1.3, color="#ea4335", label="аналитика (гипербола)")
    ax.plot(0, 0, "o", ms=14, color="#34a853", label="Юл")
    ax.set_aspect("equal", adjustable="box")
    bb = b_hyp(rp, tr.v_inf_j) / 1e3
    ax.set_xlim(-6 * bb, 4 * bb); ax.set_ylim(-6 * bb, 6 * bb)
    ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
    ax.set_xlabel("x, км"); ax.set_ylabel("y, км")
    ax.set_title("Гипербола пролёта Юла (h_p = 20 000 км):\nчисленное интегрирование RK4 и аналитическая коника")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    rr = np.linspace(R_JOOL + ATM_JOOL, 40 * R_JOOL, 400)
    ax.plot((rr - R_JOOL) / 1000, [math.degrees(delta_hyp(r, tr.v_inf_j)) for r in rr],
            color="#1a73e8", lw=1.8, label="асимптотическая формула δ = 2·arcsin(1/e)")
    ax.plot((rp - R_JOOL) / 1000, math.degrees(d_rk4), "o", color="#ea4335", ms=8,
            label=f"RK4 в пределах 0.95 SOI: {math.degrees(d_rk4):.2f}°")
    ax.plot((rp - R_JOOL) / 1000, d_asym, "x", color="#34a853", ms=10, mew=2,
            label=f"асимптотика на бесконечности: {d_asym:.2f}°")
    ax.set_xscale("log")
    ax.set_xlabel("высота перицентра h_p, км"); ax.set_ylabel("угол поворота δ, °")
    err_asym = abs(math.degrees(d_rk4) - d_asym)
    err_fin = abs(math.degrees(d_rk4) - d_fin_an)
    ax.set_title(f"Верификация численной схемы:\n"
                 f"RK4 = {math.degrees(d_rk4):.2f}°, асимптотика = {d_asym:.2f}° (Δ = {err_asym:.2f}°)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_verification.png"))
    plt.close(fig)
    return r_min, math.degrees(d_rk4), d_fin_an, d_asym, err_asym, err_fin


def fig_oberth(tr: Transfer):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    dlt = np.linspace(0, 180, 600)
    v_out = np.sqrt(tr.v_jool ** 2 + tr.v_inf_j ** 2
                    - 2 * tr.v_jool * tr.v_inf_j * np.cos(np.radians(dlt)))
    ax.plot(dlt, v_out, color="#1a73e8", lw=2, label="|v_вых| после пролёта")
    ax.axhline(v_esc(A_JOOL, MU_SUN), color="#9334e6", ls="--", lw=1.6,
               label="вторая космическая Кербола (5839 м/с)")
    ax.axhline(tr.v_jool, color="#9aa0a6", ls=":", lw=1.2, label="скорость Юла (4129 м/с)")
    for h in (200, 5000, 50_000):
        d = math.degrees(delta_hyp(R_JOOL + h * 1e3, tr.v_inf_j))
        vv = float(np.linalg.norm(post_flyby(R_JOOL + h * 1e3, tr.v_inf_j, +1, tr.v_jool)[1]))
        ax.plot(d, vv, "o", ms=8, label=f"h_p = {h:,} км".replace(",", " "))
    ax.set_xlabel("угол поворота δ, °")
    ax.set_ylabel("гелиоцентрическая скорость после пролёта, м/с")
    ax.set_title("«Треугольник скоростей» гравитационного манёвра:\nмодуль v∞ неизменен, поворачивается только вектор")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    hs2 = np.linspace(200, 40_000, 300)
    dvs = [oberth_dv(R_JOOL + h * 1e3, tr.v_inf_j, delta_hyp(R_JOOL + h * 1e3, tr.v_inf_j), tr.v_jool)[1]
           for h in hs2]
    ax.plot(hs2, dvs, color="#1a73e8", lw=2)
    ax.fill_between(hs2, dvs, color="#1a73e8", alpha=0.12)
    ax.axvline(ATM_JOOL / 1000, color="#fbbc04", lw=4, alpha=0.5)
    ax.text(ATM_JOOL / 1000 * 1.3, 900, "граница атмосферы\n(200 км)", fontsize=8, color="#b06000")
    ax.set_xlabel("высота перицентра h_p, км")
    ax.set_ylabel("импульс Δv для ухода из системы, м/с")
    ax.set_title("Эффект Оберта: импульс в низком перицентре\n(58.9 м/с при h_p = 200 км) закрывает недостаток поворота")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_oberth.png"))
    plt.close(fig)


def fig_scaling(tr: Transfer, real: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax = axes[0]
    rr = np.linspace(1.05, 12, 500)
    ax.plot(rr, [math.degrees(delta_hyp(r * R_JOOL, tr.v_inf_j)) for r in rr],
            lw=2, color="#1a73e8", label="Юл (система Кербола), v∞ = 1756 м/с")
    ax.plot(rr, [math.degrees(delta_hyp(r * real["R_km"], real["v_inf_km_s"], real["mu"])) for r in rr],
            lw=2, color="#ea4335", label=f"Юпитер («Вояджер-2»), v∞ = {real['v_inf_km_s']:.2f} км/с")
    ax.plot(real["rp_over_R"], real["delta_asymptotic"], "*", ms=16, color="#ea4335",
            label=f"асимптота модели: {real['delta_asymptotic']:.2f}°")
    ax.plot(real["rp_over_R"], real["delta_observed"], "o", ms=8, mfc="white", mec="#ea4335", mew=2,
            label=f"на r ≈ 60 R_J: JPL {real['delta_observed']:.2f}° / модель {real['delta_model_at_r']:.2f}°")
    ax.set_xlabel("безразмерный перицентр r_p / R_планеты")
    ax.set_ylabel("угол поворота δ, °")
    ax.set_ylim(60, 185)
    ax.set_title("Универсальная кривая δ = 2·arcsin(1/e):\nединый физический закон для модели и реального космоса")
    ax.legend(fontsize=7.8, loc="lower left")

    ax = axes[1]
    names = ["Юл:\nдо пролёта", "Юл:\nпосле", "Юпитер:\nдо пролёта", "Юпитер:\nпосле"]
    v_jool_before = tr.v_arr
    v_jool_after = float(np.linalg.norm(post_flyby(R_JOOL + 200e3, tr.v_inf_j, +1, tr.v_jool)[1]))
    vals = [v_jool_before / 1000, v_jool_after / 1000, real["v_before"], real["v_after"]]
    cols = ["#8ab4f8", "#1a73e8", "#f28b82", "#ea4335"]
    bars = ax.bar(names, vals, color=cols, alpha=0.9, width=0.58)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("гелиоцентрическая скорость, км/с")
    ax.set_ylim(0, 24)
    ax.set_title("Прирост гелиоцентрической скорости за счёт пращи:\nЮл (h_p = 200 км) и «Вояджер-2» у Юпитера (09.07.1979)")
    ax.text(0.03, 0.94,
            f"Юл: +{vals[1]-vals[0]:.2f} км/с (без затрат топлива)\n"
            f"«Вояджер-2»: +{vals[3]-vals[2]:.2f} км/с (JPL Horizons)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#f1f3f4", ec="#dadce0"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_scaling.png"))
    plt.close(fig)


# --- 7. Основной сценарий расчёта ----------------------------------------------------

def main():
    tr = launch_window()
    print("=" * 78)
    print("1. ГОМАНОВСКИЙ ПЕРЕЛЁТ КЕРБИН → ЮЛ")
    print("=" * 78)
    print(f"  большая полуось a_t        : {tr.a_t/1e9:8.2f} млн км")
    print(f"  время перелёта             : {tr.tof/DAY:8.1f} сут ({tr.tof/KSP_DAY:.0f} керб. сут)")
    print(f"  v∞ отлёта от Кербина       : {tr.v_inf_k:8.1f} м/с")
    print(f"  Δv старта с орбиты 100 км  : {tr.dv_tmi:8.1f} м/с")
    print(f"  орбитальная скорость Юла   : {tr.v_jool:8.1f} м/с")
    print(f"  гелиоцентр. скорость входа : {tr.v_arr:8.1f} м/с")
    print(f"  избыточная скорость v∞     : {tr.v_inf_j:8.1f} м/с")
    print(f"  угол опережения Юла        : {tr.phase_deg:8.2f}°")

    print("\n" + "=" * 78)
    print("2. ГРАВИТАЦИОННЫЙ МАНЁВР У ЮЛА")
    print("=" * 78)
    print(f"  {'h_p, км':>9} {'e':>8} {'δ, °':>8} {'b, тыс.км':>11} {'v_вых, м/с':>11} {'афелий, млн км':>15}")
    for h in (200, 1_000, 5_000, 20_000, 35_096, 100_000, 500_000):
        rp = R_JOOL + h * 1e3
        _, v_out = post_flyby(rp, tr.v_inf_j, +1, tr.v_jool)
        el = helio_elements(np.array([A_JOOL, 0.0]), v_out)
        print(f"  {h:9d} {e_hyp(rp, tr.v_inf_j):8.4f} {math.degrees(delta_hyp(rp, tr.v_inf_j)):8.2f} "
              f"{b_hyp(rp, tr.v_inf_j)/1e3:11.0f} {np.linalg.norm(v_out):11.1f} {el['r_apo']/1e9:15.2f}")

    rp_eeloo = rp_for_apo(A_EELOO * (1 + E_EELOO), tr.v_inf_j, +1, tr.v_jool)
    print(f"\n  Бесплатный заброс на афелий орбиты Эилу ({A_EELOO*(1+E_EELOO)/1e9:.1f} млн км):")
    print(f"    высота перицентра h_p = {(rp_eeloo - R_JOOL)/1e3:.0f} км, "
          f"угол поворота δ = {math.degrees(delta_hyp(rp_eeloo, tr.v_inf_j)):.2f}°")

    v_esc_helio = v_esc(A_JOOL, MU_SUN)
    need_delta = math.degrees(math.acos((tr.v_jool ** 2 + tr.v_inf_j ** 2 - v_esc_helio ** 2)
                                        / (2 * tr.v_jool * tr.v_inf_j)))
    d_max = math.degrees(delta_hyp(R_JOOL + ATM_JOOL, tr.v_inf_j))
    vinf2, dv_ob = oberth_dv(R_JOOL + ATM_JOOL, tr.v_inf_j, math.radians(d_max), tr.v_jool)
    print(f"\n  Условие ухода из системы Кербола (v_esc = {v_esc_helio:.1f} м/с):")
    print(f"    нужен поворот δ ≥ {need_delta:.2f}°, максимум у границы атмосферы δ_max = {d_max:.2f}°")
    print(f"    эффект Оберта при h_p = 200 км: импульс Δv = {dv_ob:.1f} м/с (v∞: {tr.v_inf_j:.0f} → {vinf2:.0f} м/с)")

    r_min, d_rk4, d_fin_an, d_asym, err_asym, err_fin = fig_verify(tr)
    print("\n" + "=" * 78)
    print("3. ВЕРИФИКАЦИЯ МОДЕЛИ")
    print("=" * 78)
    print(f"  Уровень 1 — RK4 против аналитики (h_p = 20 000 км, r = 0.95 SOI):")
    print(f"    δ (RK4) = {d_rk4:.2f}°,  δ (аналитика на 0.95 SOI) = {d_fin_an:.2f}°,  "
          f"δ (асимптота) = {d_asym:.2f}°")
    print(f"    r_p (RK4) = {r_min/1e3:.1f} км при заданном 26 000 км")

    real = check_real_world()
    print(f"\n  Уровень 2 — NASA JPL Horizons («Вояджер-2» у Юпитера, 09.07.1979):")
    print(f"    r_p = {real['r_p_km']:.0f} км ({real['rp_over_R']:.2f} R_J),  v∞ = {real['v_inf_km_s']:.3f} км/с")
    print(f"    e = {real['e']:.4f},  δ (асимптота) = {real['delta_asymptotic']:.2f}°")
    print(f"    угол на r ≈ 60 R_J: модель {real['delta_model_at_r']:.2f}° vs JPL {real['delta_observed']:.2f}° "
          f"(расхождение {real['err_deg']:.2f}°)")
    print(f"    гелиоцентрическая скорость: {real['v_before']:.2f} → {real['v_after']:.2f} км/с "
          f"(+{real['dv_actual']:.2f} км/с)")

    fig_trajectory(tr, rp_eeloo, R_JOOL + ATM_JOOL)
    fig_design_map(tr)
    fig_oberth(tr)
    fig_scaling(tr, real)
    # Удалим старое имя fig6_scaling.png, если оно осталось
    old_fig6 = os.path.join(OUT, "fig6_scaling.png")
    if os.path.exists(old_fig6):
        os.remove(old_fig6)

    summary = dict(
        transfer=asdict(tr),
        flyby=dict(
            h_p_eeloo_km=(rp_eeloo - R_JOOL) / 1e3,
            delta_eeloo_deg=math.degrees(delta_hyp(rp_eeloo, tr.v_inf_j)),
            delta_max_deg=d_max,
            need_delta_escape_deg=need_delta,
            v_escape_m_s=v_esc_helio,
            oberth_dv_m_s=dv_ob,
        ),
        rk4_check=dict(
            delta_rk4_deg=d_rk4,
            delta_analytic_finite_deg=d_fin_an,
            delta_asymptotic_deg=d_asym,
            err_vs_asymptotic_deg=err_asym,
            err_vs_finite_deg=err_fin,
            rp_rk4_km=r_min / 1e3,
        ),
        voyager2_jpl_check=real,
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты и 5 графиков сохранены в: {OUT}")


if __name__ == "__main__":
    main()
