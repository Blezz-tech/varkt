#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Контрольный расчёт пролёта у Юла.

Считает то, что составляет ядро проекта:
  1) эксцентриситет, угол поворота и прицельный параметр гиперболы;
  2) скорость в перицентре;
  3) гелиоцентрическую скорость после пролёта (векторно, «треугольник скоростей»);
  4) параметры послепролётной орбиты: a, e, афелий;
  5) обратную задачу: какая высота перицентра даёт нужный афелий;
  6) недостающий поворот для ухода из системы и импульс Оберта;
  7) геометрию пролёта r(ν) и время в сфере действия.

Запуск:
    python3 src/hand_flyby.py           # таблицы чисел
    python3 src/hand_flyby.py --fig     # + рисунок out/fig8_flyby_geometry.png

Зависимости: только math (matplotlib — лишь для рисунка).
Все числа сверены с основным расчётом src/sim.py и с отчётом report/report.typ.
"""

import math
import sys

# ---------------------------------------------------------------------------
# 1. Исходные данные (KSP 1.12, stock) — те же, что в отчёте, таблица 1
# ---------------------------------------------------------------------------
MU_SUN = 1.1723328e18       # м³/с²  — гравитационный параметр Кербола
MU_JOOL = 2.82528e14        # м³/с²  — гравитационный параметр Юла
R_JOOL = 6.0e6              # м      — радиус Юла (6000 км)
A_JOOL = 68.773560320e9     # м      — радиус орбиты Юла
SOI_JOOL = 2.4559852e9      # м      — радиус сферы действия Юла
ATM_JOOL = 200e3            # м      — высота атмосферы Юла (ниже — нельзя)
V_INF = 1756.2300957315365  # м/с    — скорость прилёта, из гомановского перелёта

V_JOOL = math.sqrt(MU_SUN / A_JOOL)      # 4128.7 м/с — скорость Юла по орбите
V_ESC = math.sqrt(2 * MU_SUN / A_JOOL)   # 5838.9 м/с — скорость ухода из системы


# ---------------------------------------------------------------------------
# 2. Гиперболический пролёт: шаги 1–4 модели
# ---------------------------------------------------------------------------
def flyby(h_p, v_inf=V_INF, mu=MU_JOOL, r_pl=R_JOOL):
    """Высота перицентра -> (r_p, e, δ, b)."""
    r_p = r_pl + h_p
    e = 1 + r_p * v_inf ** 2 / mu              # (2.5)  e = 1 + r_p·v∞²/μ
    delta = 2 * math.asin(1 / e)               # (2.7)  δ = 2·arcsin(1/e)
    b = r_p * math.sqrt((e + 1) / (e - 1))     # (2.8)  b = r_p·√((e+1)/(e−1))
    return r_p, e, delta, b


def v_pericenter(r_p, v_inf=V_INF, mu=MU_JOOL):
    """Скорость в перицентре: v² = v∞² + 2μ/r_p (vis-viva)."""
    return math.sqrt(v_inf ** 2 + 2 * mu / r_p)


# ---------------------------------------------------------------------------
# 3. Что получается после пролёта: шаги 5–7 модели
# ---------------------------------------------------------------------------
def post_flyby(h_p, v_inf=V_INF):
    """
    Векторный расчёт послепролётной гелиоцентрической орбиты.

    До пролёта аппарат медленнее Юла, поэтому v∞ направлена против его
    движения: v∞_in = (0, −v∞). После поворота на δ:
        v∞_out = ( v∞·sin δ, −v∞·cos δ )
    Полная скорость:  v_out = (0, v_j) + v∞_out
    """
    _, e, delta, _ = flyby(h_p, v_inf)
    v_inf_x = v_inf * math.sin(delta)
    v_inf_y = -v_inf * math.cos(delta)

    v_x, v_y = v_inf_x, V_JOOL + v_inf_y
    v = math.hypot(v_x, v_y)                   # |v_out| — «треугольник скоростей»

    r = A_JOOL
    eps = v * v / 2 - MU_SUN / r               # энергия
    h = r * v_y                                # момент импульса (r вдоль x)
    if eps >= 0:
        return dict(v=v, v_r=v_x, v_t=v_y, a=math.inf, e=math.inf, apo=math.inf)
    a = -MU_SUN / (2 * eps)
    e_hel = math.sqrt(1 + 2 * eps * h * h / MU_SUN ** 2)
    return dict(v=v, v_r=v_x, v_t=v_y, a=a, e=e_hel, apo=a * (1 + e_hel))


def h_p_for_apoapsis(apo_target, v_inf=V_INF):
    """Обратная задача: высота перицентра, при которой афелий = apo_target."""
    def f(h_p):
        return post_flyby(h_p, v_inf)["apo"] - apo_target
    lo, hi = ATM_JOOL, 200 * R_JOOL        # 200 км ... 2 радиуса Юла за SOI
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 4. Уход из системы и эффект Оберта: шаги 8–9 модели
# ---------------------------------------------------------------------------
def delta_needed_for_escape():
    """Какой поворот нужен, чтобы гелиоцентрическая скорость достигла v_ESC."""
    c = (V_JOOL ** 2 + V_INF ** 2 - V_ESC ** 2) / (2 * V_JOOL * V_INF)
    return math.acos(c)


def oberth(h_p=ATM_JOOL, v_inf=V_INF):
    """Импульс в перицентре, добирающий поворот до условия ухода."""
    _, _, delta, _ = flyby(h_p, v_inf)
    need = delta_needed_for_escape()
    bb = -2 * V_JOOL * math.cos(delta)
    cc = V_JOOL ** 2 - V_ESC ** 2
    v_inf2 = (-bb + math.sqrt(bb * bb - 4 * cc)) / 2     # нужная v∞' (из треугольника)
    r_p = R_JOOL + h_p
    dv = math.sqrt(v_inf2 ** 2 + 2 * MU_JOOL / r_p) - math.sqrt(v_inf ** 2 + 2 * MU_JOOL / r_p)
    return need, v_inf2, dv


# ---------------------------------------------------------------------------
# 5. Геометрия пролёта: r(ν), скорость, время от перицентра
# ---------------------------------------------------------------------------
def flyby_geometry(h_p=ATM_JOOL, nus=(0, 30, 60, 90, 120, 150)):
    """Полярная гипербола r = p/(1+e·cos ν) и уравнение Кеплера для времени."""
    r_p, e, delta, b = flyby(h_p)
    p = r_p * (1 + e)                       # p = h²/μ = r_p(1+e)
    a_abs = MU_JOOL / V_INF ** 2            # |a| гиперболы
    rows = []
    for nu_deg in nus:
        nu = math.radians(nu_deg)
        r = p / (1 + e * math.cos(nu))
        ch = (r / a_abs + 1) / e            # r = |a|(e·cosh H − 1)
        H = math.acosh(max(ch, 1.0))
        t = math.sqrt(a_abs ** 3 / MU_JOOL) * (e * math.sinh(H) - H)
        v = math.sqrt(MU_JOOL * (2 / r + 1 / a_abs))
        rows.append((nu_deg, r, v, t / 86400))
    # время от перицентра до границы сферы действия (0.95 SOI)
    r_soi = 0.95 * SOI_JOOL
    H = math.acosh((r_soi / a_abs + 1) / e)
    t_soi = math.sqrt(a_abs ** 3 / MU_JOOL) * (e * math.sinh(H) - H) / 86400
    nu_inf = math.degrees(math.acos(-1 / e))
    return dict(e=e, delta=delta, b=b, p=p, a_abs=a_abs, rows=rows,
                t_soi=t_soi, nu_inf=nu_inf, v_p=v_pericenter(r_p))


# ---------------------------------------------------------------------------
# 6. Печать результатов
# ---------------------------------------------------------------------------
def main():
    line = "=" * 78
    print(line)
    print("КОНТРОЛЬНЫЙ РАСЧЁТ ПРОЛЁТА У ЮЛА")
    print(line)
    print(f"Скорость Юла на орбите       v_Юла = {V_JOOL:.1f} м/с")
    print(f"Скорость ухода из системы    v_ухода = {V_ESC:.1f} м/с")
    print(f"Скорость прилёта             v∞ = {V_INF:.1f} м/с")
    print(f"Граница сферы действия       SOI = {SOI_JOOL/1e6:.0f} тыс. км")

    print("\n" + line)
    print("1. ПРОЛЁТ: e, δ, b и скорость в перицентре")
    print(line)
    print(f"{'h_p, км':>10} {'e':>8} {'δ, °':>8} {'b, км':>11} {'v_p, м/с':>10}")
    for h in (200e3, 5000e3, 20000e3, 35095.5e3, 100000e3):
        r_p, e, delta, b = flyby(h)
        print(f"{h/1e3:10.0f} {e:8.4f} {math.degrees(delta):8.2f} "
              f"{b/1e3:11.0f} {v_pericenter(r_p):10.0f}")
    print("  δ падает с ростом h_p: ближе к планете — сильнее поворот,")
    print("  но выше 200 км нельзя — атмосфера.")

    print("\n" + line)
    print("2. ПОСЛЕ ПРОЛЁТА: гелиоцентрическая орбита")
    print(line)
    print(f"{'h_p, км':>10} {'v_out, м/с':>11} {'v_r':>7} {'v_t':>7} "
          f"{'a, млн км':>10} {'e_гел':>6} {'афелий, млн км':>15}")
    for h in (200e3, 20000e3, 100000e3):
        s = post_flyby(h)
        print(f"{h/1e3:10.0f} {s['v']:11.1f} {s['v_r']:7.0f} {s['v_t']:7.0f} "
              f"{s['a']/1e9:10.2f} {s['e']:6.3f} {s['apo']/1e9:15.2f}")
    print(f"  v_out меньше v_ухода ({V_ESC:.0f} м/с) — система остаётся связанной.")

    print("\n" + line)
    print("3. ОБРАТНАЯ ЗАДАЧА: куда надо пролететь, чтобы выйти на афелий Эилу")
    print(line)
    apo_eeloo = 113.5497132e9   # a(1+e) Эилу: 90.11882e9 * 1.26
    h_target = h_p_for_apoapsis(apo_eeloo)
    _, e_t, d_t, b_t = flyby(h_target)
    print(f"  афелий Эилу = {apo_eeloo/1e9:.1f} млн км  →  h_p ≈ {h_target/1e3:.0f} км, "
          f"δ = {math.degrees(d_t):.2f}°, b = {b_t/1e3:.0f} тыс. км")
    print("  Пролёт «бесплатно» забрасывает аппарат в 1.65 раза дальше орбиты Юла.")

    print("\n" + line)
    print("4. ПОЧЕМУ НЕЛЬЗЯ УЙТИ ПРОЛЁТОМ: эффект Оберта")
    print(line)
    need, v_inf2, dv = oberth()
    _, _, d_max, _ = flyby(ATM_JOOL)
    print(f"  нужно  δ ≥ {math.degrees(need):.2f}°")
    print(f"  максимум δ = {math.degrees(d_max):.2f}° при h_p = 200 км  →  не хватает "
          f"{math.degrees(need - d_max):.2f}°")
    print(f"  импульс в перицентре: v∞ {V_INF:.0f} → {v_inf2:.0f} м/с, Δv = {dv:.1f} м/с")

    g = flyby_geometry()
    print("\n" + line)
    print("5. ГЕОМЕТРИЯ: как именно пролетает тело")
    print(line)
    print(f"  e = {g['e']:.4f},  p = {g['p']/1e3:.0f} км,  "
          f"ν∞ = arccos(−1/e) = {g['nu_inf']:.1f}°,  b = {g['b']/1e3:.0f} км")
    print(f"{'ν, °':>6} {'r, км':>12} {'v, м/с':>9} {'t, сут':>9}")
    for nu, r, v, t in g["rows"]:
        print(f"{nu:6.0f} {r/1e3:12.0f} {v:9.0f} {t:9.2f}")
    print(f"  от перицентра до границы сферы действия: {g['t_soi']:.1f} сут "
          f"(весь пролёт ≈ {2*g['t_soi']:.0f} сут)")
    print("  при ν = 90° расстояние равно p — удобная проверка на калькуляторе.")

    if "--fig" in sys.argv:
        make_figure()


# ---------------------------------------------------------------------------
# 7. Рисунок: траектория пролёта
# ---------------------------------------------------------------------------
def make_figure(path="out/fig8_flyby_geometry.png"):
    """Рисунок: гипербола пролёта, асимптоты, прицельный параметр b, угол δ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Arc

    r_p, e, delta, b = flyby(ATM_JOOL)
    p = r_p * (1 + e)                  # p = r_p(1+e), км
    a_abs = MU_JOOL / V_INF ** 2       # |a| = 91 600 км
    nu_inf = math.acos(-1 / e)         # 159.5 град
    c_x = -a_abs * e / 1e6             # центр гиперболы, тыс. км
    slope = b / a_abs                  # наклон асимптот

    nu = np.linspace(-nu_inf * 0.999, nu_inf * 0.999, 20000)
    r = p / (1 + e * np.cos(nu))
    keep = r < 165e6
    x = r[keep] * np.cos(nu[keep]) / 1e6
    y = r[keep] * np.sin(nu[keep]) / 1e6

    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=150)
    ax.plot(x, y, lw=2.4, color="#1f4e79", zorder=5, label="траектория аппарата")

    xs = np.linspace(-168, 28, 10)
    for sgn in (+1, -1):
        ax.plot(xs, sgn * slope * (xs - c_x), ls="--", lw=0.9, color="#888", zorder=1)
    ax.annotate("асимптоты", (-128, -52), fontsize=8, color="#888")

    # b — перпендикуляр из фокуса (центра планеты) к асимптоте
    k = slope
    xp = (k * (0 + k * c_x)) / (1 + k * k)
    yp = k * (xp - c_x)
    ax.plot([0, xp], [0, yp], lw=1.6, color="#b22222", zorder=6)
    ax.annotate(f"b = {b/1e6:.1f} тыс. км", (xp - 30, yp + 7), fontsize=9, color="#b22222")

    th = np.linspace(0, 2 * np.pi, 400)
    ax.fill(R_JOOL / 1e6 * np.cos(th), R_JOOL / 1e6 * np.sin(th),
            color="#c8a165", zorder=7, label="Юл (R = 6000 км)")
    ax.plot([0], [0], "+", ms=10, color="k", zorder=8)
    ax.annotate("перицентр,\nh_p = 200 км", (3.5, -13), fontsize=8)

    ax.annotate("вход в сферу действия", (-118, 78), fontsize=8, color="#1f4e79")
    ax.annotate("выход из сферы действия", (-58, -84), fontsize=8, color="#1f4e79")

    # врезка: векторы v∞ до и после
    ins = ax.inset_axes([0.04, 0.04, 0.30, 0.30])
    ins.annotate("", xy=(0, -1.25), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", lw=1.8, color="#2e7d32"))
    ins.annotate("", xy=(1.25 * math.sin(delta), -1.25 * math.cos(delta)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="->", lw=1.8, color="#2e7d32"))
    ins.add_patch(Arc((0, 0), 1.0, 1.0, theta1=-90, theta2=-90 + math.degrees(delta),
                      lw=1.0, color="#2e7d32"))
    ins.text(-0.06, -1.45, "v∞ вх.", fontsize=7, ha="center", color="#2e7d32")
    ins.text(1.05 * math.sin(delta) + 0.08, -1.05 * math.cos(delta) + 0.15,
             "v∞ вых.", fontsize=7, color="#2e7d32")
    ins.text(-0.95, 0.55, "δ = 138.98°", fontsize=8, color="#2e7d32")
    ins.set_xlim(-1.4, 1.6)
    ins.set_ylim(-1.6, 1.1)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    ins.set_title("поворот скорости", fontsize=8)
    for sp in ins.spines.values():
        sp.set_color("#2e7d32")
        sp.set_linewidth(0.8)

    ax.set_xlim(-160, 34)
    ax.set_ylim(-105, 105)
    ax.set_xlabel("x, тыс. км")
    ax.set_ylabel("y, тыс. км")
    ax.set_title("Как пролетает тело: гипербола у Юла, e = 1.068")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    print(f"\nрисунок сохранён: {path}")


if __name__ == "__main__":
    main()
