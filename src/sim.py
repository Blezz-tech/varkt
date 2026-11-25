"""
Физико-математическая модель гравитационного манёвра у планеты
(перелёт Земля -> Юпитер в Солнечной системе и проверка формул на справочных
данных реального пролёта «Вояджера-2» у Юпитера, NASA, 9 июля 1979 г.).

Модель: плоская (2D) — нулевое приближение; некомпланарность орбит Земли и
Юпитера (i = 1.303° к эклиптике, NASA) учтена аналитическими поправками
(функция coplanarity): векторный расчёт в худшем случае (встреча в узле),
наклонение выходящей асимптоты, азимутальный угол beta, отклонение положения
Юпитера от эклиптики против размера СОИ. Метод точечных сфер действия
(patched conics).
Участки:
  1) гомановский перелёт Земля -> Юпитер в поле Солнца;
  2) гиперболический пролёт в сфере действия Юпитера (аналитика + цикл RK4);
  3) послепролётная гелиоцентрическая орбита («треугольник скоростей»);
  4) условие ухода из системы и импульс в перицентре (эффект Оберта);
  5) сравнение модели с фактическим полётом «Вояджера-2» (NASA) и проверка формул
     на табличных параметрах его пролёта у Юпитера.

Единицы: СИ (для справочного блока «Вояджера-2» — км и км/с).
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

# --- 1. Физические константы Солнечной системы (СИ, JPL/NASA) ------------------------
MU_SUN = 1.32712440018e20
MU_EARTH, R_EARTH, A_EARTH = 3.986004418e14, 6_371_000.0, 149_597_870_700.0
MU_JUPITER, R_JUPITER, A_JUPITER = 1.26686534e17, 71_492_000.0, 778_479_000_000.0
SOI_JUPITER, ATM_JUPITER = 48_214_000_000.0, 200_000.0
A_SATURN, E_SATURN = 1_433_530_000_000.0, 0.0565
# Наклонение орбиты Юпитера к плоскости эклиптики (орбита Земли), рад
# (NASA Planetary Fact Sheet: Inclination to Ecliptic = 1.303°)
I_JUPITER_ECL = math.radians(1.303)
DAY = 86_400.0

# Табличные данные NASA по пролёту «Вояджера-2» у Юпитера (9 июля 1979 г.)
VOYAGER2 = dict(
    mu_jup=1.266865e8,   # гравитационный параметр Юпитера, км³/с²
    R_jup=71_492.0,      # экваториальный радиус Юпитера, км
    r_p=721_895.0,       # радиус перицентра пролёта (10.10 R_Ю), км
    v_inf=7.60,          # гиперболическая избыточная скорость, км/с
    v_jup=13.06,         # орбитальная скорость Юпитера, км/с
    v_in_helio=9.86,     # гелиоцентрическая скорость до пролёта, км/с
    v_out_nasa=20.03,    # фактическая гелиоцентрическая скорость после пролёта, км/с
)


def period(a: float, mu: float) -> float:
    return 2.0 * math.pi * math.sqrt(a ** 3 / mu)


def v_circ(r: float, mu: float) -> float:
    return math.sqrt(mu / r)


def v_esc(r: float, mu: float) -> float:
    return math.sqrt(2.0 * mu / r)


# --- 2. Гомановский перелёт Земля -> Юпитер ---------------------------------------------

@dataclass
class Transfer:
    r1: float
    a_t: float
    tof: float
    dv_tmi: float
    v_inf_k: float
    v_arr: float
    v_jup: float
    v_inf_j: float
    phase_deg: float


def launch_window(r1: float = R_EARTH + 100_000.0) -> Transfer:
    """Расчёт полуэллипса Гомана от орбиты Земли до орбиты Юпитера."""
    a_t = 0.5 * (A_EARTH + A_JUPITER)
    tof = math.pi * math.sqrt(a_t ** 3 / MU_SUN)
    v_c1 = v_circ(A_EARTH, MU_SUN)
    v_inf_k = math.sqrt(MU_SUN * (2.0 / A_EARTH - 1.0 / a_t)) - v_c1
    v_arr = math.sqrt(MU_SUN * (2.0 / A_JUPITER - 1.0 / a_t))
    v_j = v_circ(A_JUPITER, MU_SUN)
    v_inf_j = abs(v_j - v_arr)
    v_per = math.sqrt(v_inf_k ** 2 + 2.0 * MU_EARTH / r1)
    dv = v_per - v_circ(r1, MU_EARTH)
    n_j = 2.0 * math.pi / period(A_JUPITER, MU_SUN)
    phase = (math.pi - n_j * tof + math.pi) % (2.0 * math.pi) - math.pi
    return Transfer(r1, a_t, tof, dv, v_inf_k, v_arr, v_j, v_inf_j, math.degrees(phase))


# --- 3. Гиперболический пролёт в сфере действия Юпитера ---------------------------------

def e_hyp(rp: float, vinf: float, mu: float = MU_JUPITER) -> float:
    """Эксцентриситет пролётной гиперболы: e = 1 + r_p * v_inf^2 / mu."""
    return 1.0 + rp * vinf ** 2 / mu


def delta_hyp(rp: float, vinf: float, mu: float = MU_JUPITER) -> float:
    """Угол поворота вектора скорости: delta = 2 * arcsin(1/e)."""
    return 2.0 * math.asin(1.0 / e_hyp(rp, vinf, mu))


def b_hyp(rp: float, vinf: float, mu: float = MU_JUPITER) -> float:
    """Прицельный параметр: b = r_p * sqrt((e + 1) / (e - 1)).

    При vinf -> 0 гипербола вырождается в параболу (e -> 1) и b -> inf.
    """
    if vinf <= 0.0:
        return math.inf
    e = e_hyp(rp, vinf, mu)
    return rp * math.sqrt((e + 1.0) / (e - 1.0))


def rk4_step(r: np.ndarray, v: np.ndarray, dt: float, mu: float = MU_JUPITER):
    """Один классический шаг метода Рунге—Кутты 4-го порядка для r'' = -mu * r / |r|^3."""
    def acc(pos):
        return -mu * pos / (np.linalg.norm(pos) ** 3)

    k1_r, k1_v = v, acc(r)
    k2_r, k2_v = v + 0.5 * dt * k1_v, acc(r + 0.5 * dt * k1_r)
    k3_r, k3_v = v + 0.5 * dt * k2_v, acc(r + 0.5 * dt * k2_r)
    k4_r, k4_v = v + dt * k3_v, acc(r + dt * k3_r)
    r_next = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return r_next, v_next


def propagate_rk4_from_pericenter(rp: float, vinf: float, dt: float = 20.0,
                                  r_limit: float = 4.5e8, mu: float = MU_JUPITER):
    """
    Простейшая численная проверка гиперболы с постоянным шагом dt:
    стартуем в перицентре r0 = (rp, 0), v0 = (0, v_p) и интегрируем вперёд и назад по времени.
    """
    vp = math.sqrt(vinf ** 2 + 2.0 * mu / rp)
    r0, v0 = np.array([rp, 0.0]), np.array([0.0, vp])

    # Вперёд от перицентра (+dt) и назад от перицентра (-dt)
    fwd, r, v = [r0], r0.copy(), v0.copy()
    while np.linalg.norm(r) < r_limit:
        r, v = rk4_step(r, v, +dt, mu)
        fwd.append(r)
    v_out = v

    bwd, r, v = [], r0.copy(), v0.copy()
    while np.linalg.norm(r) < r_limit:
        r, v = rk4_step(r, v, -dt, mu)
        bwd.append(r)
    v_in = v

    traj = np.array(bwd[::-1] + fwd)
    cos_d = np.clip(np.dot(v_in, v_out) / (np.linalg.norm(v_in) * np.linalg.norm(v_out)), -1.0, 1.0)
    return traj, math.degrees(math.acos(float(cos_d)))


# --- 4. Послепролётная гелиоцентрическая орбита и эффект Оберта ----------------------

def turn_vinf(vinf_in: np.ndarray, delta: float, side: int) -> np.ndarray:
    """Поворот вектора v_inf на угол delta (side = +1 позади планеты — разгон)."""
    c, s = math.cos(side * delta), math.sin(side * delta)
    return np.array([[c, -s], [s, c]]) @ vinf_in


def post_flyby(rp: float, vinf: float, side: int, v_j: float):
    """Треугольник скоростей: v_out = v_J + v_inf_out."""
    vinf_in = np.array([0.0, -vinf])
    vinf_out = turn_vinf(vinf_in, delta_hyp(rp, vinf), side)
    return vinf_out, np.array([0.0, v_j]) + vinf_out


def helio_elements(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_SUN) -> dict:
    """Большая полуось, эксцентриситет и афелий гелиоцентрической орбиты по r и v."""
    r, v = float(np.linalg.norm(r_vec)), float(np.linalg.norm(v_vec))
    eps = 0.5 * v * v - mu / r
    h = float(r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0])
    a = -mu / (2.0 * eps) if eps < 0 else math.inf
    e_vec = ((v * v - mu / r) * r_vec - float(np.dot(r_vec, v_vec)) * v_vec) / mu
    e = float(np.linalg.norm(e_vec))
    r_per = h * h / mu / (1.0 + e)
    return dict(eps=eps, a=a, e=e, r_per=r_per,
                r_apo=a * (1.0 + e) if math.isfinite(a) else math.inf)


def rp_for_apo(apo_target: float, vinf: float, side: int, v_j: float,
               r_helio: float = A_JUPITER) -> float | None:
    """Подбор радиуса перицентра пролёта под заданный афелий методом деления отрезка пополам."""
    def f(rp):
        _, v_out = post_flyby(rp, vinf, side, v_j)
        return helio_elements(np.array([r_helio, 0.0]), v_out)["r_apo"] - apo_target

    lo, hi = R_JUPITER + ATM_JUPITER + 10_000, 200 * R_JUPITER
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def oberth_dv(rp: float, vinf: float, v_j: float,
              mu: float = MU_JUPITER) -> tuple[float, float]:
    """Импульс Delta_v в перицентре гиперболы для достижения второй космической скорости.

    Самосогласованный расчёт: импульс увеличивает v_inf до v_inf', что меняет
    эксцентриситет e' = 1 + r_p * v_inf'^2 / mu и угол поворота
    delta' = 2 * arcsin(1 / e'). Требуемая v_inf' — наименьший корень уравнения
    |v_вых(v_inf', delta')| = v_esc (треугольник скоростей с обновлённым delta').
    Фиксирование delta на доимпульсном значении (квадратичная формула) занижает
    Delta_v: при h_p = 200 км даёт 4.29 против самосогласованных 4.77 м/с (~10 %);
    при другом запасе поворота расхождение может достигать порядка.
    Корень ищется сканированием (первое пересечение) + делением отрезка пополам,
    т.к. |v_вых(v_inf')| немонотонен.
    """
    v_esc_helio = v_esc(A_JUPITER, MU_SUN)

    def v_out_of(vinf2: float) -> float:
        d = delta_hyp(rp, vinf2, mu)
        return math.sqrt(v_j ** 2 + vinf2 ** 2 - 2.0 * v_j * vinf2 * math.cos(d))

    def gap(vinf2: float) -> float:
        return v_out_of(vinf2) - v_esc_helio

    if gap(vinf) >= 0.0:
        return vinf, 0.0
    # Верхняя граница с gap > 0 (при v_inf' -> inf выходная скорость -> inf).
    hi = max(2.0 * vinf, vinf + 1000.0)
    while gap(hi) <= 0.0 and hi < 1e7:
        hi = vinf + 2.0 * (hi - vinf)
    # Первое пересечение нуля (минимальный v_inf' => минимальный Delta_v).
    n_scan = 500
    prev = vinf
    bracket = None
    for i in range(1, n_scan + 1):
        cur = vinf + (hi - vinf) * i / n_scan
        if gap(cur) >= 0.0:
            bracket = (prev, cur)
            break
        prev = cur
    if bracket is None:  # pragma: no cover — недостижимо при hi с gap > 0
        bracket = (vinf, hi)
    lo, hi_b = bracket
    for _ in range(80):
        mid = 0.5 * (lo + hi_b)
        if gap(mid) >= 0.0:
            hi_b = mid
        else:
            lo = mid
    vinf2 = 0.5 * (lo + hi_b)
    dv = math.sqrt(vinf2 ** 2 + 2.0 * mu / rp) - math.sqrt(vinf ** 2 + 2.0 * mu / rp)
    return vinf2, dv


def coplanarity(rp: float, vinf: float, v_j: float, v_arr: float,
                mu: float = MU_JUPITER) -> dict:
    """Поправки за некомпланарность орбит Земли и Юпитера (i = 1.303°).

    Плоскость перелёта = эклиптика (орбита Земли, i = 0). Орбита Юпитера
    наклонена к эклиптике на i_J = 1.303° (NASA). Разбираются два граничных
    случая, покрывающие любую фазу встречи:

    1) Встреча в узле: положение Юпитера на эклиптике, но вектор его скорости
       наклонён к плоскости перелёта на epsilon = i_J. Тогда:
       - |v∞| в системе Юпитера растёт (векторная разность v_прил - v_J3);
       - плоскость пролёта (содержит v∞ и радиус перицентра, лежащий
         в эклиптике — прицеливание сохраняется в проекции на эклиптику)
         наклонена к орбитальной плоскости Юпитера на азимутальный угол beta;
       - гелиоцентрическая скорость после пролёта и наклонение выходящей
         асимптоты считаются векторно (поворот Родригеса в плоскости пролёта).
    2) Встреча в максимальной широте: Юпитер отклонён от эклиптики не более
       чем на a_J * sin(i_J) — это отклонение целиком поглощается сферой
       действия (r_SOI); при этом v_J лежит в эклиптике и поправок к |v∞|,
       e, delta нет вовсе.

    Поправки малы (сотые-десятые процентов), но допущение компланарности
    снято: его влияние ограничено и вычислено. Импульс Оберта зависит только
    от |v∞| и r_p в системе планеты, поэтому меняется лишь на величину
    поправки к |v∞|.
    """
    eps = I_JUPITER_ECL
    # --- Случай 1: встреча в узле, v_J наклонена на epsilon -----------------
    v_j3 = np.array([0.0, v_j * math.cos(eps), v_j * math.sin(eps)])
    v_in3 = np.array([0.0, v_arr, 0.0]) - v_j3           # v∞, система Юпитера
    vinf3 = float(np.linalg.norm(v_in3))
    d3 = delta_hyp(rp, vinf3, mu)  # уже в радианах
    # Направление радиуса перицентра — в эклиптике, перпендикулярно проекции v∞.
    r_hat = np.cross(np.array([0.0, 0.0, 1.0]), v_in3)
    r_hat /= np.linalg.norm(r_hat)
    # Нормаль плоскости пролёта: m = v∞ × r_p (в плоском пределе — +z).
    m = np.cross(v_in3, r_hat)
    m /= np.linalg.norm(m)
    # Поворот v∞ на delta3 вокруг m (формула Родригеса).
    c, s = math.cos(d3), math.sin(d3)
    v_out3 = (v_in3 * c + np.cross(m, v_in3) * s
              + m * float(np.dot(m, v_in3)) * (1.0 - c))
    v_helio3 = v_j3 + v_out3
    # Плоский эталон (та же геометрия при epsilon = 0).
    _, v_helio2 = post_flyby(rp, vinf, +1, v_j)
    v2 = float(np.linalg.norm(v_helio2))
    v3 = float(np.linalg.norm(v_helio3))
    # Азимутальный угол beta: между v∞ и орбитальной плоскостью Юпитера.
    # В узле радиус-вектор Юпитера лежит в эклиптике (в кадре post_flyby — вдоль +x).
    n_j = np.cross(np.array([1.0, 0.0, 0.0]), v_j3 / v_j)
    beta = math.degrees(math.asin(abs(float(np.dot(v_in3, n_j))) / vinf3))
    # Наклонение выходящей асимптоты к эклиптике.
    gamma = math.degrees(math.asin(abs(float(v_helio3[2])) / v3))
    # Импульс Оберта с поправленным |v∞| (величина в системе планеты).
    _, dv_ob3 = oberth_dv(rp, vinf3, v_j, mu)
    _, dv_ob2 = oberth_dv(rp, vinf, v_j, mu)
    # Граничный случай 2: отклонение положения Юпитера от эклиптики в узле-вне.
    miss_max = A_JUPITER * math.sin(eps)
    return dict(
        incl_ecl_deg=math.degrees(eps),
        v_inf_planar_m_s=vinf, v_inf_3d_m_s=vinf3, d_v_inf_m_s=vinf3 - vinf,
        delta_planar_deg=math.degrees(delta_hyp(rp, vinf, mu)),
        delta_3d_deg=math.degrees(d3),
        v_out_planar_m_s=v2, v_out_3d_m_s=v3, d_v_out_m_s=v3 - v2,
        outplane_v_out_m_s=float(abs(v_helio3[2])),
        gamma_out_deg=gamma, beta_max_deg=beta,
        oberth_dv_planar_m_s=dv_ob2, oberth_dv_3d_m_s=dv_ob3,
        node_miss_max_m=miss_max, soi_m=SOI_JUPITER,
        soi_covers_miss=bool(miss_max < SOI_JUPITER),
    )


# --- 5. Проверка формул по справочным параметрам «Вояджера-2» ------------------------

def check_voyager2() -> dict:
    """
    Подстановка справочных табличных чисел пролёта «Вояджера-2» у Юпитера (NASA, 09.07.1979)
    в выведенные формулы гиперболы и треугольника скоростей.
    """
    mu, rp, vinf = VOYAGER2["mu_jup"], VOYAGER2["r_p"], VOYAGER2["v_inf"]
    vj, vin = VOYAGER2["v_jup"], VOYAGER2["v_in_helio"]

    e = e_hyp(rp, vinf, mu)
    delta = delta_hyp(rp, vinf, mu)
    # Угол входа между векторами v_Юпитера и v_inf до пролёта по теореме косинусов:
    cos_phi_in = (vin ** 2 - vj ** 2 - vinf ** 2) / (2.0 * vj * vinf)
    phi_in = math.acos(cos_phi_in)
    # После поворота вектора v_inf к направлению движения Юпитера на угол delta:
    phi_out = phi_in - delta
    v_out_model = math.sqrt(vj ** 2 + vinf ** 2 + 2.0 * vj * vinf * math.cos(phi_out))

    return dict(
        r_p_km=rp,
        rp_over_R=rp / VOYAGER2["R_jup"],
        v_inf_km_s=vinf,
        e=e,
        delta_deg=math.degrees(delta),
        v_in_km_s=vin,
        v_out_model_km_s=v_out_model,
        v_out_nasa_km_s=VOYAGER2["v_out_nasa"],
        dv_nasa_km_s=VOYAGER2["v_out_nasa"] - vin,
        rel_err_percent=abs(v_out_model - VOYAGER2["v_out_nasa"]) / VOYAGER2["v_out_nasa"] * 100.0,
    )


# --- 6. Построение рисунков для отчёта -----------------------------------------------

def _circle(r: float, n: int = 600):
    th = np.linspace(0, 2 * np.pi, n)
    return r * np.cos(th), r * np.sin(th)


def _ellipse(a: float, e: float, omega: float = 0.0, n: int = 1200):
    th = np.linspace(0, 2 * np.pi, n)
    b = a * math.sqrt(max(1.0 - e * e, 1e-12))
    x, y = a * np.cos(th) - a * e, b * np.sin(th)
    c, s = math.cos(omega), math.sin(omega)
    return c * x - s * y, s * x + c * y


def _conic_branch(r0: np.ndarray, v0: np.ndarray, mu: float = MU_SUN, n: int = 2500):
    rn = float(np.linalg.norm(r0))
    h = float(r0[0] * v0[1] - r0[1] * v0[0])
    e_vec = ((float(v0 @ v0) - mu / rn) * r0 - float(r0 @ v0) * v0) / mu
    e = float(np.linalg.norm(e_vec))
    p = h * h / mu
    phi_e = math.atan2(float(e_vec[1]), float(e_vec[0]))
    nu = np.linspace(0, 2 * math.pi, n)
    r = p / (1.0 + e * np.cos(nu))
    return r * np.cos(phi_e + nu), r * np.sin(phi_e + nu)


def fig_trajectory(tr: Transfer, rp_saturn: float, rp_low: float):
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    ax.set_aspect("equal")
    for r, c, lab in ((A_EARTH, "#8ab4f8", "орбита Земли"),
                      (A_JUPITER, "#34a853", "орбита Юпитера")):
        x, y = _circle(r)
        ax.plot(x / 1e9, y / 1e9, color=c, lw=1.2, label=lab)
    x, y = _ellipse(A_SATURN, E_SATURN, math.radians(260))
    ax.plot(x / 1e9, y / 1e9, color="#9aa0a6", lw=1.1, ls="--", label="орбита Сатурна")

    th_tr = np.linspace(0, math.pi, 800)
    e_tr = (A_JUPITER - A_EARTH) / (A_JUPITER + A_EARTH)
    r_tr = tr.a_t * (1.0 - e_tr ** 2) / (1.0 + e_tr * np.cos(th_tr))
    ax.plot(r_tr * np.cos(th_tr) / 1e9, r_tr * np.sin(th_tr) / 1e9,
            color="#ea4335", lw=2.3, label="гомановский перелёт Земля → Юпитер")

    r_meet = np.array([-A_JUPITER, 0.0])
    _, v_low = post_flyby(rp_low, tr.v_inf_j, +1, tr.v_jup)
    apo_low = helio_elements(r_meet, v_low)["r_apo"] / 1e9
    h_sat_km = (rp_saturn - R_JUPITER) / 1e3
    for rp, color, lab in ((rp_low, "#e37400",
                            f"после пролёта h_p = 200 км (афелий {apo_low:,.0f} млн км)".replace(",", " ")),
                           (rp_saturn, "#9334e6",
                            f"после пролёта h_p = {h_sat_km:,.0f} км (к афелию Сатурна)".replace(",", " "))):
        _, v_out_std = post_flyby(rp, tr.v_inf_j, +1, tr.v_jup)
        xh, yh = _conic_branch(r_meet, -v_out_std, mu=MU_SUN)
        ax.plot(xh / 1e9, yh / 1e9, color=color, lw=1.9, label=lab)

    ax.plot(0, 0, "o", ms=14, color="#fbbc04", label="Солнце")
    ax.plot(-A_JUPITER / 1e9, 0, "o", ms=9, color="#34a853", label="точка встречи с Юпитером")
    ax.plot(A_EARTH / 1e9, 0, "o", ms=7, color="#4285f4", label="старт с Земли")
    ax.set_xlim(-1620, 1720)
    ax.set_ylim(-1660, 1660)
    ax.set_xlabel("x, млн км")
    ax.set_ylabel("y, млн км")
    ax.set_title("Схема миссии (2D): перелёт Земля → Юпитер,\nгравитационный пролёт и послепролётные орбиты")
    ax.legend(loc="upper right", fontsize=7.6)
    fig.savefig(os.path.join(OUT, "fig1_trajectory.png"))
    plt.close(fig)


def fig_design_map(tr: Transfer, hs=np.linspace(200, 2e6, 900)):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    ax = axes[0]
    apo = np.array([
        helio_elements(np.array([A_JUPITER, 0.0]), post_flyby(R_JUPITER + h * 1000, tr.v_inf_j, +1, tr.v_jup)[1])["r_apo"]
        for h in hs
    ])
    ax.plot(hs, apo / 1e9, color="#1a73e8", lw=2)
    ax.fill_between(hs, 0, apo / 1e9, color="#1a73e8", alpha=0.10)
    ax.axhline(A_SATURN / 1e9, color="#9334e6", ls="--", lw=1.5, label="большая полуось Сатурна")
    ax.axhline(A_SATURN * (1.0 + E_SATURN) / 1e9, color="#34a853", ls="--", lw=1.5,
               label=f"афелий Сатурна ({A_SATURN * (1.0 + E_SATURN) / 1e9:,.0f} млн км)".replace(",", " "))
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(150, 3e6)
    ax.set_xticks([200, 1000, 10_000, 100_000, 1_000_000])
    ax.set_xticklabels(["200", "1k", "10k", "100k", "1M"])
    ax.set_xlabel("высота перицентра пролёта h_p, км")
    ax.set_ylabel("афелий орбиты после пролёта, млн км")
    ax.set_title("Эффективность гравитационного манёвра:\nчем ближе пролёт, тем сильнее поворот и выше афелий")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    need_esc = np.array([
        oberth_dv(R_JUPITER + h * 1000, tr.v_inf_j, tr.v_jup)[1]
        for h in hs
    ])
    ax.plot(hs, need_esc, color="#e37400", lw=2)
    ax.fill_between(hs, 0, need_esc, color="#e37400", alpha=0.12)
    ax.axvspan(0, ATM_JUPITER / 1000, color="#fbbc04", alpha=0.35)
    ax.text(ATM_JUPITER / 1000 * 1.15, float(np.max(need_esc)) * 0.55, "атмосфера\nЮпитера",
            fontsize=8, color="#b06000")
    ax.set_xscale("log")
    ax.set_xlim(150, 3e6)
    ax.set_xticks([200, 1000, 10_000, 100_000, 1_000_000])
    ax.set_xticklabels(["200", "1k", "10k", "100k", "1M"])
    ax.set_xlabel("высота перицентра пролёта h_p, км")
    ax.set_ylabel("импульс Δv в перицентре для ухода из системы, м/с")
    ax.set_title("Цена ухода из Солнечной системы:\nчем глубже перицентр, тем меньше нужный Δv")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_design_map.png"))
    plt.close(fig)


def fig_verify(tr: Transfer):
    rp = R_JUPITER + 20_000_000.0
    r_lim = 0.95 * SOI_JUPITER
    traj_rk4, d_rk4 = propagate_rk4_from_pericenter(rp, tr.v_inf_j, dt=20.0, r_limit=r_lim)
    d_asym = math.degrees(delta_hyp(rp, tr.v_inf_j))

    # Аналитическая кривая гиперболы r(nu) = p / (1 + e * cos(nu))
    e = e_hyp(rp, tr.v_inf_j)
    p = rp * (1.0 + e)
    nu_max = math.acos((p / r_lim - 1.0) / e)
    nu = np.linspace(-nu_max, nu_max, 1000)
    rr = p / (1.0 + e * np.cos(nu))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.plot(traj_rk4[:, 0] / 1e3, traj_rk4[:, 1] / 1e3, lw=2.0, color="#1a73e8",
            label="численно (метод RK4, шаг 20 с)")
    ax.plot(rr * np.cos(nu) / 1e3, rr * np.sin(nu) / 1e3, "--", lw=1.4, color="#ea4335",
            label="формула коники r(ν) = p / (1 + e cos ν)")
    ax.plot(0, 0, "o", ms=13, color="#34a853", label="Юпитер")
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(style="sci", scilimits=(-3, 3), axis="both")
    ax.set_xlabel("x, км"); ax.set_ylabel("y, км")
    ax.set_title("Гипербола пролёта Юпитера (h_p = 20 000 км):\nсовпадение численного расчёта RK4 и формулы коники")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    hs = np.linspace(ATM_JUPITER, 40 * R_JUPITER, 400)
    ax.plot(hs / 1000, [math.degrees(delta_hyp(R_JUPITER + h, tr.v_inf_j)) for h in hs],
            color="#1a73e8", lw=1.9, label="формула δ = 2·arcsin(1/e)")
    for h_test in (200e3, 5_000e3, 20_000e3, 100_000e3):
        _, d_num = propagate_rk4_from_pericenter(R_JUPITER + h_test, tr.v_inf_j, dt=60.0, r_limit=0.95 * SOI_JUPITER)
        ax.plot(h_test / 1000, d_num, "o", color="#ea4335", ms=7,
                label="точки численного расчёта RK4" if h_test == 200e3 else None)
    ax.set_xscale("log")
    ax.set_xlabel("высота перицентра h_p, км"); ax.set_ylabel("угол поворота δ, °")
    ax.set_title("Угол поворота вектора скорости δ(h_p):\nаналитическая кривая и контрольные точки RK4")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_verification.png"))
    plt.close(fig)
    return d_rk4, d_asym


def fig_oberth(tr: Transfer):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    dlt = np.linspace(0, 180, 600)
    v_out = np.sqrt(tr.v_jup ** 2 + tr.v_inf_j ** 2
                    - 2.0 * tr.v_jup * tr.v_inf_j * np.cos(np.radians(dlt)))
    ax.plot(dlt, v_out, color="#1a73e8", lw=2, label="|v_вых| после пролёта")
    v_esc_h = v_esc(A_JUPITER, MU_SUN)
    ax.axhline(v_esc_h, color="#9334e6", ls="--", lw=1.6,
               label=f"вторая космическая Солнца ({v_esc_h:.0f} м/с)")
    ax.axhline(tr.v_jup, color="#9aa0a6", ls=":", lw=1.2,
               label=f"скорость Юпитера ({tr.v_jup:.0f} м/с)")
    for h in (200, 5000, 50_000):
        d = math.degrees(delta_hyp(R_JUPITER + h * 1e3, tr.v_inf_j))
        vv = float(np.linalg.norm(post_flyby(R_JUPITER + h * 1e3, tr.v_inf_j, +1, tr.v_jup)[1]))
        ax.plot(d, vv, "o", ms=8, label=f"h_p = {h:,} км".replace(",", " "))
    ax.set_xlabel("угол поворота δ, °")
    ax.set_ylabel("гелиоцентрическая скорость после пролёта, м/с")
    ax.set_title("«Треугольник скоростей» гравитационного манёвра:\nмодуль v∞ неизменен, поворачивается только вектор")
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    hs2 = np.linspace(200, 40_000, 300)
    dvs = [oberth_dv(R_JUPITER + h * 1e3, tr.v_inf_j, tr.v_jup)[1]
           for h in hs2]
    ax.plot(hs2, dvs, color="#1a73e8", lw=2)
    ax.fill_between(hs2, dvs, color="#1a73e8", alpha=0.12)
    ax.axvline(ATM_JUPITER / 1000, color="#fbbc04", lw=4, alpha=0.5)
    ax.text(ATM_JUPITER / 1000 * 1.3, float(np.max(dvs)) * 0.55, "граница атмосферы\n(200 км)",
            fontsize=8, color="#b06000")
    ax.set_xlabel("высота перицентра h_p, км")
    ax.set_ylabel("импульс Δv для ухода из системы, м/с")
    ax.set_title(f"Эффект Оберта: импульс в низком перицентре\n({dvs[0]:.1f} м/с при h_p = 200 км) закрывает недостаток поворота")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_oberth.png"))
    plt.close(fig)


def fig_scaling(tr: Transfer, voy: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    ax = axes[0]
    rr = np.linspace(1.05, 12, 500)
    ax.plot(rr, [math.degrees(delta_hyp(r * R_JUPITER, tr.v_inf_j)) for r in rr],
            lw=2, color="#1a73e8",
            label=f"модель Гомана: Юпитер, v∞ = {tr.v_inf_j:.0f} м/с")
    ax.plot(rr, [math.degrees(delta_hyp(r * VOYAGER2["R_jup"], VOYAGER2["v_inf"], VOYAGER2["mu_jup"])) for r in rr],
            lw=2, color="#ea4335",
            label=f"пролёт «Вояджера-2»: Юпитер, v∞ = {VOYAGER2['v_inf']:.2f} км/с")
    ax.plot(voy["rp_over_R"], voy["delta_deg"], "*", ms=15, color="#ea4335",
            label=f"точка пролёта «Вояджера-2» (10.1 R_Ю, δ = {voy['delta_deg']:.1f}°)")
    ax.set_xlabel("безразмерный перицентр r_p / R_планеты")
    ax.set_ylabel("угол поворота δ, °")
    ax.set_ylim(60, 185)
    ax.set_title("Универсальная кривая δ = 2·arcsin(1/e):\nодна формула для модели и реального пролёта Юпитера")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    names = ["До пролёта\n(NASA)", "После пролёта\n(расчёт модели)", "После пролёта\n(факт NASA)"]
    vals = [voy["v_in_km_s"], voy["v_out_model_km_s"], voy["v_out_nasa_km_s"]]
    cols = ["#8ab4f8", "#1a73e8", "#34a853"]
    bars = ax.bar(names, vals, color=cols, alpha=0.9, width=0.52)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f} км/с", ha="center", fontsize=9.5)
    ax.set_ylabel("гелиоцентрическая скорость, км/с")
    ax.set_ylim(0, 26)
    ax.set_title("Проверка по миссии «Вояджер-2» у Юпитера (09.07.1979):\nвыходная скорость по нашей 2D-модели и по факту NASA")
    ax.text(0.03, 0.94,
            f"Расчёт 2D-модели: {voy['v_out_model_km_s']:.2f} км/с\n"
            f"Факт NASA: {voy['v_out_nasa_km_s']:.2f} км/с (погрешность {voy['rel_err_percent']:.1f} %)",
            transform=ax.transAxes, va="top", fontsize=8.8,
            bbox=dict(boxstyle="round", fc="#f1f3f4", ec="#dadce0"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_scaling.png"))
    plt.close(fig)


# --- 7. Основной сценарий расчёта ----------------------------------------------------

def main():
    tr = launch_window()
    rp_saturn = rp_for_apo(A_SATURN * (1.0 + E_SATURN), tr.v_inf_j, +1, tr.v_jup)
    h_sat = int(round((rp_saturn - R_JUPITER) / 1e3))
    print("=" * 78)
    print("1. ГОМАНОВСКИЙ ПЕРЕЛЁТ ЗЕМЛЯ → ЮПИТЕР")
    print("=" * 78)
    print(f"  большая полуось a_t        : {tr.a_t/1e9:8.2f} млн км")
    print(f"  время перелёта             : {tr.tof/DAY:8.1f} сут ({tr.tof/DAY/365.25:.2f} г)")
    print(f"  Δv старта с орбиты 100 км  : {tr.dv_tmi:8.1f} м/с")
    print(f"  орбитальная скорость Юпитера: {tr.v_jup:8.1f} м/с")
    print(f"  гелиоцентр. скорость входа : {tr.v_arr:8.1f} м/с")
    print(f"  избыточная скорость v∞     : {tr.v_inf_j:8.1f} м/с")
    print(f"  угол опережения Юпитера     : {tr.phase_deg:8.2f}°")

    print("\n" + "=" * 78)
    print("2. ГРАВИТАЦИОННЫЙ МАНЁВР У ЮПИТЕРА")
    print("=" * 78)
    print(f"  {'h_p, км':>9} {'e':>8} {'δ, °':>8} {'b, км':>11} {'v_вых, м/с':>11} {'афелий, млн км':>15}")
    for h in (200, 1_000, 5_000, 20_000, 100_000, 500_000, h_sat):
        rp = R_JUPITER + h * 1e3
        _, v_out = post_flyby(rp, tr.v_inf_j, +1, tr.v_jup)
        el = helio_elements(np.array([A_JUPITER, 0.0]), v_out)
        print(f"  {h:9d} {e_hyp(rp, tr.v_inf_j):8.4f} {math.degrees(delta_hyp(rp, tr.v_inf_j)):8.2f} "
              f"{b_hyp(rp, tr.v_inf_j)/1e3:11.0f} {np.linalg.norm(v_out):11.1f} {el['r_apo']/1e9:15.2f}")

    v_esc_helio = v_esc(A_JUPITER, MU_SUN)
    need_delta = math.degrees(math.acos((tr.v_jup ** 2 + tr.v_inf_j ** 2 - v_esc_helio ** 2)
                                        / (2.0 * tr.v_jup * tr.v_inf_j)))
    d_max = math.degrees(delta_hyp(R_JUPITER + ATM_JUPITER, tr.v_inf_j))
    vinf2, dv_ob = oberth_dv(R_JUPITER + ATM_JUPITER, tr.v_inf_j, tr.v_jup)
    print(f"\n  Условие ухода из Солнечной системы (v_esc = {v_esc_helio:.1f} м/с):")
    print(f"    нужен поворот δ ≥ {need_delta:.2f}°, максимум у границы атмосферы δ_max = {d_max:.2f}°")
    print(f"    эффект Оберта при h_p = 200 км: импульс Δv = {dv_ob:.1f} м/с (v∞: {tr.v_inf_j:.0f} → {vinf2:.0f} м/с)")

    cop = coplanarity(R_JUPITER + ATM_JUPITER, tr.v_inf_j, tr.v_jup, tr.v_arr)
    print(f"\n  Некомпланарность орбит (i Юпитера = {cop['incl_ecl_deg']:.3f}° к эклиптике):")
    print(f"    встреча в узле:  v∞ {cop['v_inf_planar_m_s']:.1f} → {cop['v_inf_3d_m_s']:.1f} м/с "
          f"(Δ = {cop['d_v_inf_m_s']:+.1f}),  δ_max {cop['delta_planar_deg']:.2f} → {cop['delta_3d_deg']:.2f}°")
    print(f"    v_вых {cop['v_out_planar_m_s']:.1f} → {cop['v_out_3d_m_s']:.1f} м/с "
          f"(Δ = {cop['d_v_out_m_s']:+.1f}),  наклонение выходящей асимптоты {cop['gamma_out_deg']:.2f}°")
    print(f"    азимутальный угол β ≤ {cop['beta_max_deg']:.2f}°,  Δv Оберта "
          f"{cop['oberth_dv_planar_m_s']:.2f} → {cop['oberth_dv_3d_m_s']:.2f} м/с")
    print(f"    отклонение Юпитера от эклиптики ≤ {cop['node_miss_max_m'] / 1e9:.1f} млн км "
          f"при r_SOI = {SOI_JUPITER / 1e9:.1f} млн км: "
          f"{'поглощается СОИ' if cop['soi_covers_miss'] else 'ВНЕ СОИ!'}")

    _, d_num_soi = propagate_rk4_from_pericenter(R_JUPITER + 20_000e3, tr.v_inf_j, dt=20.0, r_limit=0.95 * SOI_JUPITER)
    d_asym = math.degrees(delta_hyp(R_JUPITER + 20_000e3, tr.v_inf_j))
    voy = check_voyager2()
    _, v_out_passive = post_flyby(R_JUPITER + ATM_JUPITER, tr.v_inf_j, +1, tr.v_jup)
    v_out_passive_kms = float(np.linalg.norm(v_out_passive)) / 1e3

    print("\n" + "=" * 78)
    print("3. ПРОВЕРКА МОДЕЛИ (RK4 + ТАБЛИЧНЫЕ ДАННЫЕ «ВОЯДЖЕРА-2»)")
    print("=" * 78)
    print(f"  Численный расчёт RK4 (шаг 20 с, h_p = 20 000 км):")
    print(f"    δ (RK4 в пределах СОИ) = {d_num_soi:.2f}°,  δ (формула 2·arcsin(1/e)) = {d_asym:.2f}°")
    print(f"  Подстановка табличных параметров «Вояджера-2» у Юпитера (NASA, 09.07.1979):")
    print(f"    r_p = {voy['r_p_km']:.0f} км ({voy['rp_over_R']:.2f} R_Ю),  v∞ = {voy['v_inf_km_s']:.2f} км/с")
    print(f"    e = {voy['e']:.4f},  расчётный угол поворота δ = {voy['delta_deg']:.2f}°")
    print(f"    выходная гелиоцентрическая скорость: модель {voy['v_out_model_km_s']:.2f} км/с vs факт NASA "
          f"{voy['v_out_nasa_km_s']:.2f} км/с (погрешность 2D-модели {voy['rel_err_percent']:.1f} %)")

    fig_trajectory(tr, rp_saturn, R_JUPITER + ATM_JUPITER)
    fig_design_map(tr)
    fig_verify(tr)
    fig_oberth(tr)
    fig_scaling(tr, voy)

    summary = dict(
        transfer=asdict(tr),
        flyby=dict(
            h_p_saturn_km=(rp_saturn - R_JUPITER) / 1e3,
            delta_saturn_deg=math.degrees(delta_hyp(rp_saturn, tr.v_inf_j)),
            delta_max_deg=d_max,
            need_delta_escape_deg=need_delta,
            v_escape_m_s=v_esc_helio,
            oberth_dv_m_s=dv_ob,
        ),
        comparison=dict(
            fact_launch="1977-08-20", fact_jupiter="1979-07-09",
            fact_tof_days=689, model_tof_days=tr.tof / DAY,
            fact_v_inf_km_s=VOYAGER2["v_inf"], model_v_inf_km_s=tr.v_inf_j / 1e3,
            fact_v_in_helio_km_s=VOYAGER2["v_in_helio"], model_v_in_helio_km_s=tr.v_arr / 1e3,
            fact_v_out_km_s=VOYAGER2["v_out_nasa"],
            model_v_out_passive_km_s=v_out_passive_kms,
            fact_delta_deg=check_voyager2()["delta_deg"],
            model_delta_max_deg=d_max,
            model_oberth_dv_m_s=dv_ob,
        ),
        rk4_check=dict(delta_rk4_soi_deg=d_num_soi, delta_formula_deg=d_asym),
        voyager2_check=voy,
        coplanarity=cop,
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nРезультаты и 5 графиков сохранены в: {OUT}")


if __name__ == "__main__":
    main()
