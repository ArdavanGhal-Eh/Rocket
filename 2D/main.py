"""
====================================================================================================
شبیه‌سازی کامل، فیزیکی و دوبعدی (2D) رهگیری و برخورد موشک به جنگنده با مقادیر واقعی و صنعتی
2D Full Physical & Industrial Aerospace Simulation: Missile-Target Interception (2D TPN vs 2D APN)
====================================================================================================

مبانی تئوری و دینامیکی الگوریتم‌های ناوبری در صفحه دوبعدی:
----------------------------------------------------------------------------------------------------
۱. ناوبری تناسبی دوبعدی حقیقی (2D True Proportional Navigation - 2D TPN):
   در صفحه درگیری دو بعدی (x, y):
       R_vec = r_T - r_M = [Rx, Ry]
       V_rel = v_T - v_M = [V_rel_x, V_rel_y]
       Range R = |R_vec| = sqrt(Rx^2 + Ry^2)
       R_hat = R_vec / R
       n_hat = [-R_hat_y, R_hat_x]
   سرعت همگرایی (Closing Velocity):
       V_c = - dR/dt = - (R_vec . V_rel) / R
   نرخ چرخش زاویه‌ای خط دید در صفحه:
       lambda_dot = (Rx * V_rel_y - Ry * V_rel_x) / R^2
   بردار شتاب فرمان ناوبری عمود بر خط دید:
       a_cmd = N * V_c * lambda_dot * n_hat

   تحلیل رفتار دینامیکی:
   در مانورهای شدید جنگنده، روش 2D TPN به دلیل تاخیر فاز کینماتیکی ناشی از انباشت خطای زاویه‌ای خط دید،
   در فاز پایانی (Endgame) با جهش شدید تقاضای شتاب مواجه شده که منجر به اشباع شتاب (35G) و افزایش خطای اصابت می‌گردد.

۲. ناوبری تناسبی ارتقایافته دوبعدی (2D Augmented Proportional Navigation - 2D APN):
   مولفه شتاب هدف عمود بر بردار خط دید:
       a_T_perp = (a_T . n_hat) * n_hat
   بردار شتاب فرمان پیش‌خور APN:
       a_cmd = N * V_c * lambda_dot * n_hat + (N / 2) * a_T_perp

   مزیت فیزیکی:
   خنثی‌سازی بلادرنگ شتاب مانور هدف در صفحه درگیری، حفظ نرخ خط دید نزدیک به صفر و جلوگیری از اشباع شتاب موشک.
====================================================================================================
"""

import argparse
import os
import sys

# Ensure script directory is on sys.path for robust standalone execution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Ensure UTF-8 stream handling on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from typing import List, Tuple
import numpy as np

from missile_sim_2d.config import MissileConfig2D, TargetConfig2D, SimulationConfig2D
from missile_sim_2d.guidance import GuidanceLaw2D
from missile_sim_2d.target import (
    BaseTargetManeuver2D,
    TARGET_SCENARIOS_2D,
    create_target_maneuver_2d,
)
from missile_sim_2d.simulator import SimulationEngine2D, SimulationResult2D
from missile_sim_2d.fitting import TrajectoryFitter2D, PolynomialFitResult2D
from missile_sim_2d.visualizer import SimulationVisualizer2D


def print_theoretical_and_physical_brief_2d():
    """چاپ گزارش جامع تئوریک و مبانی فیزیکی شبیه‌سازی دوبعدی در آغاز خروجی کنسول."""
    banner = r"""
============================================================================================================
              شبیه‌سازی صنعتی و دوبعدی هدایت موشک و درگیری با جنگنده متخاصم
                 2D INDUSTRIAL MISSILE INTERCEPTION & GUIDANCE BENCHMARK
============================================================================================================
 [1] مبانی تئوری و فرمولاسیون ریاضی هدایت دوبعدی (Planar 2D Guidance):
  - ناوبری تناسبی دوبعدی (2D TPN):
      * نرخ چرخش زاویه‌ای خط دید:  λ̇ = (Rx · V_rel_y - Ry · V_rel_x) / |R|²
      * سرعت همگرایی:              V_c = -dR/dt = -(R · V_rel) / |R|
      * بردار شتاب فرمان:          a_cmd = N · V_c · λ̇ · n̂_LOS
      * رفتار دینامیکی: تاخیر فاز کینماتیکی در مانورهای شدید هدف و تشدید شتاب در لحظات پایانی (Endgame Peak)

  - ناوبری تناسبی ارتقایافته (2D APN):
      * مولفه نرمال شتاب هدف:      a_T⊥ = (a_T · n̂_LOS) · n̂_LOS
      * بردار شتاب فرمان پیش‌خور:   a_cmd = N · V_c · λ̇ · n̂_LOS + (N/2) · a_T⊥
      * مزیت فیزیکی: خنثی‌سازی مستقیم مانور هدف، حذف اشباع شتاب 35G و اصابت دقیق میلی‌متری

 [2] مشخصات فیزیکی، آیرودینامیک و پیشران موشک:
  - شرایط جوی: ارتفاع ~5000m | چگالی هوا: rho = 0.736 kg/m³ | سرعت صوت: a = 320.5 m/s
  - جرم کل پرتاب: m0 = 85.0 kg | سوخت: 35.0 kg | جرم خشک پس از اتمام سوخت: m_dry = 50.0 kg
  - زمان سوزش موتور راکت: t_burn = 4.0 s | نرخ کاهش جرم: mdot = 8.75 kg/s
  - نیروی تراست راکت: T = 17,500 N در طول 4 ثانیه نخست (در امتداد بردار سرعت)، سپس فاز سرش آزاد با T = 0 N
  - پسا آیرودینامیکی: F_D = -0.5 · rho · |v|² · C_D · A · v̂ (قطر d = 0.127m، مساحت A = 0.0127m²، C_D = 0.40)
  - سقف شتاب جانبی مجاز: 35G (معادل 343.35 m/s²)
  - شرایط شلیک اولیه: موقعیت [0, 0] با سرعت اولیه 250 m/s در راستای خط دید اولیه (LOS)

 [3] مشخصات جنگنده متخاصم:
  - موقعیت اولیه: x0 = 6000m, y0 = 2500m (فاصله اولیه: 6500 m)
  - سرعت سیر پایدار: 300 m/s (تقریباً 0.9 ماخ)
============================================================================================================
"""
    print(banner)


def display_results_table_2d(results: List[SimulationResult2D], fits: List[PolynomialFitResult2D]):
    """چاپ جدول مقایسه‌ای نهایی نتایج شبیه‌سازی در کنسول."""
    def _law_str(l):
        if l == GuidanceLaw2D.TPN:
            return "2D TPN"
        elif l == GuidanceLaw2D.APN:
            return "2D APN"
        return "2D Pure Pursuit"

    header = (
        f"{'شاخص عملکرد / الگوریتم':<38} | "
        + " | ".join([f"{_law_str(r.law):^22}" for r in results])
    )
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print("                    جدول مقایسه‌ای نتایج عملکرد و شاخص‌های فیزیکی اصابت (2D)")
    print("=" * len(header))
    print(header)
    print(separator)

    print(f"{'کمینه فاصله اصابت (Miss Distance - m)':<38} | " + " | ".join([f"{r.miss_distance:^22.4f}" for r in results]))
    print(f"{'زمان پرواز تا اصابت (Flight Time - s)':<38} | " + " | ".join([f"{r.intercept_time:^22.2f}" for r in results]))
    print(f"{'سرعت در لحظه اصابت (Final Speed - m/s)':<38} | " + " | ".join([f"{r.final_speed:^22.1f}" for r in results]))
    print(f"{'عدد ماخ در لحظه اصابت (Final Mach)':<38} | " + " | ".join([f"{r.final_mach:^22.2f}" for r in results]))
    print(f"{'بیشینه شتاب جانبی (Peak Lateral G)':<38} | " + " | ".join([f"{r.peak_lateral_g:^22.2f}" for r in results]))
    print(f"{'درصد زمان اشباع شتاب (G-Saturation %)':<38} | " + " | ".join([f"{r.saturation_percentage:^22.2f}" for r in results]))
    print(f"{'مصرف انرژی کنترلی (Control Energy - m²/s³)':<38} | " + " | ".join([f"{r.total_control_energy:^22.2e}" for r in results]))
    print(f"{'شاخص برازش معادلات حرکت (Mean R²)':<38} | " + " | ".join([f"{fit.overall_mean_r2:^22.6f}" for fit in fits]))
    print("=" * len(header))


def print_parametric_equations_2d(results: List[SimulationResult2D], fits: List[PolynomialFitResult2D]):
    """نمایش معادلات صریح ریاضی استخراج‌شده حرکت موشک در صفحه دوبعدی."""
    for res, fit in zip(results, fits):
        if res.law == GuidanceLaw2D.TPN:
            law_title = "2D TPN"
        elif res.law == GuidanceLaw2D.APN:
            law_title = "2D APN"
        else:
            law_title = "2D Pure Pursuit"
        print(f"\n>>> معادلات پارامتریک حرکت موشک دوبعدی ({law_title}) در مواجهه با {res.target_name}:")
        print(fit.summary_table())


def run_pipeline_2d(
    mode: int,
    scenario_id: str,
    target_kwargs: dict = None,
    poly_degree: int = 6,
    show_plots: bool = True,
    output_dir: str = "./outputs",
) -> Tuple[List[SimulationResult2D], List[PolynomialFitResult2D]]:
    os.makedirs(output_dir, exist_ok=True)
    target_kwargs = target_kwargs or {}

    missile_cfg = MissileConfig2D()
    target_cfg = TargetConfig2D()
    sim_cfg = SimulationConfig2D()

    target_maneuver = create_target_maneuver_2d(scenario_id, target_cfg, **target_kwargs)
    engine = SimulationEngine2D(missile_cfg, target_cfg, sim_cfg)

    laws_to_run: List[GuidanceLaw2D] = []
    if mode == 1:
        laws_to_run = [GuidanceLaw2D.TPN]
    elif mode == 2:
        laws_to_run = [GuidanceLaw2D.APN]
    elif mode == 3:
        laws_to_run = [GuidanceLaw2D.PP]
    elif mode == 4:
        laws_to_run = [GuidanceLaw2D.TPN, GuidanceLaw2D.APN, GuidanceLaw2D.PP]
    else:
        raise ValueError(f"Invalid mode {mode}. Expected 1, 2, 3, or 4.")

    results: List[SimulationResult2D] = []
    for law in laws_to_run:
        print(f"[*] در حال شبیه‌سازی پرواز موشک با الگوریتم {law.value} علیه {target_maneuver.name} ...")
        res = engine.run(law, target_maneuver)
        results.append(res)
        print(
            f"    -> اصابت کامل! زمان: {res.intercept_time:.2f}s | "
            f"فاصله کمینه (Miss Distance): {res.miss_distance:.4f} m | "
            f"سرعت نهایی: {res.final_speed:.1f} m/s (ماخ {res.final_mach:.2f})"
        )

    fitter = TrajectoryFitter2D(degree=poly_degree)
    fits: List[PolynomialFitResult2D] = []
    for res in results:
        fit = fitter.fit(res.time, res.r_M)
        fits.append(fit)

    display_results_table_2d(results, fits)
    print_parametric_equations_2d(results, fits)

    viz = SimulationVisualizer2D()
    suffix = f"mode{mode}_scen{scenario_id.upper()}"
    path_2d_traj = os.path.join(output_dir, f"trajectory_planar_{suffix}.png")
    path_2d_telemetry = os.path.join(output_dir, f"telemetry_2d_{suffix}.png")
    path_html = os.path.join(output_dir, f"intercept_2d_{suffix}.html")

    print("\n[*] در حال تولید پلات مسیر دوبعدی و پنل‌های تحلیلی زمانی ...")
    viz.plot_trajectories_2d_matplotlib(
        results,
        title="2D Missile vs Maneuvering Fighter Interception",
        num_los_rays=7,
        save_path=path_2d_traj,
        show=show_plots,
    )

    viz.plot_telemetry_2d(
        results,
        title="Flight Dynamics & Guidance Telemetry (2D)",
        save_path=path_2d_telemetry,
        show=show_plots,
    )

    print("[*] در حال تولید داشبورد تعاملی دوبعدی Plotly HTML ...")
    html_file = viz.generate_interactive_plotly_2d(
        results,
        title="Interactive 2D Interception Engagement (TPN vs APN)",
        output_html=path_html,
        num_los_rays=8,
    )

    print(f"\n[+] شبیه‌سازی دوبعدی با موفقیت کامل به پایان رسید.")
    print(f"    - نمودار مسیر دوبعدی ذخیره شد در: {os.path.abspath(path_2d_traj)}")
    print(f"    - گراف‌های تحلیلی دوبعدی ذخیره شد در: {os.path.abspath(path_2d_telemetry)}")
    print(f"    - داشبورد تعاملی دوبعدی ذخیره شد در: {os.path.abspath(html_file)}")

    return results, fits


def interactive_menu_2d():
    print("\n" + "=" * 65)
    print("   منوی تعاملی انتخاب سناریو و الگوریتم‌های شبیه‌سازی دوبعدی (2D)")
    print("=" * 65)
    print(" [1] حالت ۱: شبیه‌سازی پرواز موشک با ناوبری تناسبی دوبعدی (2D TPN)")
    print(" [2] حالت ۲: شبیه‌سازی پرواز موشک با ناوبری تناسبی ارتقایافته (2D APN)")
    print(" [3] حالت ۳: شبیه‌سازی پرواز موشک با الگوریتم تعقیب محض (2D Pure Pursuit - سر به هدف)")
    print(" [4] حالت ۴: مقایسه همزمان هر ۳ الگوریتم (TPN vs APN vs Pure Pursuit) [پیش‌فرض]")

    mode_input = input("\nشماره حالت (1-4) [پیش‌فرض 4]: ").strip()
    mode = int(mode_input) if mode_input in ["1", "2", "3", "4"] else 4

    print("\nانتخاب نوع مسیر / سناریوی هدف در دوبعدی:")
    print(" [A] سناریوی الف: پرواز یکنواخت مستقیم (Straight Line)")
    print(" [B] سناریوی ب: مانور موجی/سینوسی مداوم (Sinusoidal Weave ~7.5G)")
    print(" [C] سناریوی ج: گردش با شتاب جانبی شدید دایروی (High-G Circular Turn 7.5G)")
    print(" [D] سناریوی د: مانور گریز زیگزاگی شدید (High-G 2D S-Turn 7-9G)")
    print(" [E / LINE] معادله خط هدف (Linear Equation: y = m*x + c)")
    print(" [F / PARABOLA] معادله سهمی هدف (Parabolic Path: y = a*x² + b*x + c)")
    print(" [G / CUSTOM] معادله توابع دلخواه ریاضی (Custom Parametric Functions: x(t), y(t))")

    scen_input = input("\nشناسه سناریو یا معادله [پیش‌فرض D]: ").strip().upper()
    target_kwargs = {}

    if scen_input in ["E", "LINE"]:
        scenario = "LINE"
        slope_in = input("شیب خط m [پیش‌فرض 0.4167]: ").strip()
        slope = float(slope_in) if slope_in else 0.416667
        inter_in = input("عرض از مبدأ c [پیش‌فرض 0.0]: ").strip()
        intercept = float(inter_in) if inter_in else 0.0
        target_kwargs = {"slope": slope, "intercept": intercept}
    elif scen_input in ["F", "PARABOLA"]:
        scenario = "PARABOLA"
        a_in = input("ضریب a سهمی [پیش‌فرض 0.00005]: ").strip()
        a = float(a_in) if a_in else 0.00005
        b_in = input("ضریب b سهمی [پیش‌فرض -0.2]: ").strip()
        b = float(b_in) if b_in else -0.2
        c_in = input("عدد ثابت c سهمی [پیش‌فرض 1900.0]: ").strip()
        c = float(c_in) if c_in else 1900.0
        target_kwargs = {"a": a, "b": b, "c": c}
    elif scen_input in ["G", "CUSTOM"]:
        scenario = "CUSTOM"
        x_in = input("معادله x(t) [پیش‌فرض 6000 - 250*t]: ").strip()
        y_in = input("معادله y(t) [پیش‌فرض 2500 + 400*sin(0.6*t)]: ").strip()
        x_expr = x_in if x_in else "6000 - 250*t"
        y_expr = y_in if y_in else "2500 + 400*sin(0.6*t)"
        target_kwargs = {"x_expr": x_expr, "y_expr": y_expr}
    else:
        scenario = scen_input if scen_input in ["A", "B", "C", "D"] else "D"

    return mode, scenario, target_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="2D Industrial Physical Simulation of Missile-Target Interception (TPN vs APN vs Pure Pursuit)"
    )
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4], default=None)
    parser.add_argument(
        "--target", "--scenario", dest="target", type=str, default=None,
        help="Target scenario: A, B, C, D, LINE, PARABOLA, or CUSTOM"
    )
    parser.add_argument("--custom-x", type=str, default="6000 - 250*t")
    parser.add_argument("--custom-y", type=str, default="2500 + 400*sin(0.6*t)")
    parser.add_argument("--line-slope", type=float, default=0.416667)
    parser.add_argument("--line-intercept", type=float, default=0.0)
    parser.add_argument("--parabola-a", type=float, default=0.00005)
    parser.add_argument("--parabola-b", type=float, default=-0.2)
    parser.add_argument("--parabola-c", type=float, default=1900.0)
    parser.add_argument("--poly-deg", type=int, default=6)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=str, default=os.path.join(SCRIPT_DIR, "outputs"))

    args = parser.parse_args()

    print_theoretical_and_physical_brief_2d()

    target_kwargs = {}
    if args.mode is None or args.target is None:
        if sys.stdin.isatty():
            mode, scenario, target_kwargs = interactive_menu_2d()
        else:
            mode = 4
            scenario = "D"
            print(f"[Auto-Select] Running default Mode {mode} with Scenario {scenario}.")
    else:
        mode = args.mode
        scenario = args.target.upper().strip()
        if scenario == "LINE":
            target_kwargs = {"slope": args.line_slope, "intercept": args.line_intercept}
        elif scenario == "PARABOLA":
            target_kwargs = {"a": args.parabola_a, "b": args.parabola_b, "c": args.parabola_c}
        elif scenario == "CUSTOM":
            target_kwargs = {"x_expr": args.custom_x, "y_expr": args.custom_y}

    run_pipeline_2d(
        mode=mode,
        scenario_id=scenario,
        target_kwargs=target_kwargs,
        poly_degree=args.poly_deg,
        show_plots=(not args.no_show) and sys.stdin.isatty(),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
