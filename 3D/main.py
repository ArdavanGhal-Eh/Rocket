"""
====================================================================================================
شبیه‌سازی کامل، فیزیکی و سه‌بعدی (3D) رهگیری و برخورد موشک به جنگنده با مقادیر واقعی و صنعتی
3D Full Physical & Industrial Aerospace Simulation: Missile-Target Interception (TPN vs APN)
====================================================================================================

مبانی تئوری و دینامیکی الگوریتم‌های ناوبری در فضای سه‌بعدی:
----------------------------------------------------------------------------------------------------
۱. ناوبری تناسبی سه‌بعدی حقیقی (3D True Proportional Navigation - 3D TPN):
   در فضای سه‌بعدی اقلیدسی، بردار موقعیت نسبی R و بردار سرعت نسبی V_rel عبارتند از:
       R_vec = r_T - r_M
       V_rel = v_T - v_M
   سرعت همگرایی (Closing Velocity) برابر است با نرخ کاهش فاصله:
       V_c = - d|R|/dt = - (R_vec . V_rel) / |R_vec|
   بردار سرعت زاویه‌ای چرخش خط دید در فضای سه‌بعدی (3D LOS Angular Rate Vector):
       Omega_LOS = (R_vec x V_rel) / |R_vec|^2
   بردار شتاب فرمان ناوبری عمود بر بردار خط دید R_hat:
       a_cmd = N * V_c * (Omega_LOS x R_hat)
   که در آن N ثابت ناوبری (اینجا N = 4.0) و R_hat بردار یکه خط دید است.

   تحلیل رفتار دینامیکی و تاخیر فاز (Phase Lag):
   در فضای سه‌بعدی، دوران خط دید دارای دو درجه آزادی سمتی (Azimuth) و ارتفاعی (Elevation) است.
   در مواجهه با مانورهای غیرمسطح شدید جنگنده، روش TPN تنها به نرخ دوران خط دید حاصل از جابجایی
   واکنش نشان می‌دهد. این امر موجب تاخیر فاز ذاتی و نیازمندی به شتاب‌های فوق‌العاده شدید در فاز
   پایانی (Endgame) می‌شود که غالباً منجر به اشباع شتاب مجاز (G-Saturation = 35G) و افزایش خطای فاصله می‌گردد.

۲. ناوبری تناسبی ارتقایافته سه‌بعدی (3D Augmented Proportional Navigation - 3D APN):
   برای غلبه بر تاخیر فاز و خنثی‌سازی مستقیم مانورهای هدف، بردار شتاب هدف به صورت پیش‌خور (Feedforward)
   وارد معادله هدایت می‌شود. مولفه شتاب هدف عمود بر بردار خط دید عبارت است از:
       a_T_perp = a_T - (a_T . R_hat) * R_hat
   بردار شتاب فرمان ناوبری APN به فرم زیر است:
       a_cmd = N * V_c * (Omega_LOS x R_hat) + (N / 2) * a_T_perp

   مزیت فیزیکی:
   با تزریق مستقیم جمله (N / 2) * a_T_perp در صفحه مانور سه‌بعدی، شتاب گریز جنگنده بلافاصله خنثی شده،
   نرخ چرخش خط دید در طول مسیر نزدیک به صفر نگه داشته می‌شود و مانع از اشباع شتاب در لحظات حساس اصابت می‌گردد.
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

from missile_sim.config import MissileConfig, TargetConfig, SimulationConfig
from missile_sim.guidance import GuidanceLaw
from missile_sim.target import (
    BaseTargetManeuver,
    TARGET_SCENARIOS,
    create_target_maneuver,
)
from missile_sim.missile import Missile3D
from missile_sim.simulator import SimulationEngine, SimulationResult
from missile_sim.fitting import TrajectoryFitter, PolynomialFitResult
from missile_sim.visualizer import SimulationVisualizer


def print_theoretical_and_physical_brief():
    """چاپ گزارش جامع تئوریک و مبانی فیزیکی شبیه‌سازی در آغاز خروجی کنسول."""
    banner = r"""
============================================================================================================
              شبیه‌سازی صنعتی و سه‌بعدی هدایت موشک و درگیری با جنگنده متخاصم
                 3D INDUSTRIAL MISSILE INTERCEPTION & GUIDANCE BENCHMARK
============================================================================================================
 [1] مبانی تئوری و فرمولاسیون ریاضی هدایت سه‌بعدی:
  - ناوبری تناسبی سه‌بعدی (3D TPN):
      * بردار سرعت زاویه‌ای خط دید:  Ω_LOS = (R × V_rel) / |R|²
      * سرعت همگرایی:               V_c = -dR/dt = -(R · V_rel) / |R|
      * بردار شتاب فرمان:           a_cmd = N · V_c · (Ω_LOS × R̂)
      * رفتار دینامیکی: تاخیر فاز ذاتی در مانورهای غیرمسطح هدف و تشدید شتاب در فاز پایانی (Endgame Peak)

  - ناوبری تناسبی ارتقایافته (3D APN):
      * مولفه نرمال شتاب هدف:       a_T⊥ = a_T - (a_T · R̂)R̂
      * بردار شتاب فرمان پیش‌خور:    a_cmd = N · V_c · (Ω_LOS × R̂) + (N/2) · a_T⊥
      * مزیت فیزیکی: خنثی‌سازی پیش‌دستانه مانور هدف، جلوگیری از اشباع شتاب 35G و کاهش خطای اصابت به میلی‌متر

 [2] پارامترهای فیزیکی، آیرودینامیک و پیشران موشک:
  - شرایط جوی: ارتفاع ~5000m | چگالی هوا: rho = 0.736 kg/m³ | سرعت صوت: a = 320.5 m/s
  - جرم کل پرتاب: m0 = 85.0 kg | جرم سوخت: 35.0 kg | جرم خشک پس از اتمام سوخت: m_dry = 50.0 kg
  - زمان سوزش موتور: t_burn = 4.0 s | نرخ افت جرم: mdot = 8.75 kg/s
  - نیروی تراست راکت: T = 17,500 N در طول 4 ثانیه نخست (در راستای بردار سرعت)، سپس فاز سرش آزاد با T = 0 N
  - پسا آیرودینامیکی: F_D = -0.5 · rho · |v|² · C_D · A · v̂ (قطر d = 0.127m، مساحت A = 0.0127m²، C_D = 0.40)
  - سقف شتاب جانبی مجاز: 35G (معادل 343.35 m/s²) با اشباع کروی بردار شتاب هدایت
  - شرایط شلیک اولیه: موقعیت [0, 0, 0] با سرعت اولیه 250 m/s در راستای خط دید اولیه (LOS)

 [3] مشخصات جنگنده متخاصم:
  - موقعیت اولیه: x0 = 6000m, y0 = 2500m, z0 = 5000m (فاصله اولیه: ~8201 m)
  - سرعت سیر پایدار: 300 m/s (تقریباً 0.9 ماخ)
============================================================================================================
"""
    print(banner)


def display_results_table(results: List[SimulationResult], fits: List[PolynomialFitResult]):
    """چاپ جدول مقایسه‌ای نهایی نتایج شبیه‌سازی در کنسول."""
    def _law_str(l):
        if l == GuidanceLaw.TPN:
            return "3D TPN"
        elif l == GuidanceLaw.APN:
            return "3D APN"
        return "3D Pure Pursuit"

    header = (
        f"{'شاخص عملکرد / الگوریتم':<38} | "
        + " | ".join([f"{_law_str(r.law):^22}" for r in results])
    )
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print("                    جدول مقایسه‌ای نتایج عملکرد و شاخص‌های فیزیکی اصابت")
    print("=" * len(header))
    print(header)
    print(separator)

    # Miss Distance
    row_miss = f"{'کمینه فاصله اصابت (Miss Distance - m)':<38} | " + " | ".join(
        [f"{r.miss_distance:^22.4f}" for r in results]
    )
    print(row_miss)

    # Intercept Time
    row_time = f"{'زمان پرواز تا اصابت (Flight Time - s)':<38} | " + " | ".join(
        [f"{r.intercept_time:^22.2f}" for r in results]
    )
    print(row_time)

    # Final Speed
    row_spd = f"{'سرعت در لحظه اصابت (Final Speed - m/s)':<38} | " + " | ".join(
        [f"{r.final_speed:^22.1f}" for r in results]
    )
    print(row_spd)

    # Final Mach
    row_mch = f"{'عدد ماخ در لحظه اصابت (Final Mach)':<38} | " + " | ".join(
        [f"{r.final_mach:^22.2f}" for r in results]
    )
    print(row_mch)

    # Peak Lateral G
    row_g = f"{'بیشینه شتاب جانبی (Peak Lateral G)':<38} | " + " | ".join(
        [f"{r.peak_lateral_g:^22.2f}" for r in results]
    )
    print(row_g)

    # Saturation percentage
    row_sat = f"{'درصد زمان اشباع شتاب (G-Saturation %)':<38} | " + " | ".join(
        [f"{r.saturation_percentage:^22.2f}" for r in results]
    )
    print(row_sat)

    # Control Energy
    row_energy = f"{'مصرف انرژی کنترلی (Control Energy - m²/s³)':<38} | " + " | ".join(
        [f"{r.total_control_energy:^22.2e}" for r in results]
    )
    print(row_energy)

    # Mean R^2
    row_r2 = f"{'شاخص برازش معادلات حرکت (Mean R²)':<38} | " + " | ".join(
        [f"{fit.overall_mean_r2:^22.6f}" for fit in fits]
    )
    print(row_r2)

    print("=" * len(header))


def print_parametric_equations(results: List[SimulationResult], fits: List[PolynomialFitResult]):
    """نمایش معادلات صریح ریاضی استخراج‌شده حرکت موشک در فضای سه‌بعدی."""
    for res, fit in zip(results, fits):
        if res.law == GuidanceLaw.TPN:
            law_title = "3D TPN"
        elif res.law == GuidanceLaw.APN:
            law_title = "3D APN"
        else:
            law_title = "3D Pure Pursuit"
        print(f"\n>>> معادلات پارامتریک حرکت موشک سه‌بعدی ({law_title}) در مواجهه با {res.target_name}:")
        print(fit.summary_table())


def run_pipeline(
    mode: int,
    scenario_id: str,
    target_kwargs: Optional[Dict[str, Any]] = None,
    poly_degree: int = 6,
    show_plots: bool = True,
    output_dir: str = "./outputs",
) -> Tuple[List[SimulationResult], List[PolynomialFitResult]]:
    """
    اجرای کامل پایپ‌لاین شبیه‌سازی برای سناریو و حالت انتخابی.

    Args:
        mode: 1 (TPN only), 2 (APN only), 3 (Pure Pursuit), 4 (Side-by-Side 3-Way Comparison)
        scenario_id: 'A', 'B', 'C', 'D', 'LINE', 'PARABOLA', or 'CUSTOM'
        target_kwargs: دیکشنری اختیاری پارامترهای هدف معادلاتی
        poly_degree: Order of polynomial fit (>= 5)
        show_plots: Whether to open interactive GUI windows
        output_dir: Output folder for plots and HTML dashboards
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Instantiate configuration and target maneuver
    missile_cfg = MissileConfig()
    target_cfg = TargetConfig()
    sim_cfg = SimulationConfig()

    target_kwargs = target_kwargs or {}
    target_maneuver = create_target_maneuver(scenario_id, target_cfg, **target_kwargs)

    # 2. Setup simulation engine
    engine = SimulationEngine(missile_cfg, target_cfg, sim_cfg)

    # 3. Determine guidance laws to run based on mode
    laws_to_run: List[GuidanceLaw] = []
    if mode == 1:
        laws_to_run = [GuidanceLaw.TPN]
    elif mode == 2:
        laws_to_run = [GuidanceLaw.APN]
    elif mode == 3:
        laws_to_run = [GuidanceLaw.PP]
    elif mode == 4:
        laws_to_run = [GuidanceLaw.TPN, GuidanceLaw.APN, GuidanceLaw.PP]
    else:
        raise ValueError(f"Invalid mode {mode}. Expected 1, 2, 3, or 4.")

    # 4. Execute simulations
    results: List[SimulationResult] = []
    for law in laws_to_run:
        print(f"[*] در حال شبیه‌سازی پرواز موشک با الگوریتم {law.value} علیه {target_maneuver.name} ...")
        res = engine.run(law, target_maneuver)
        results.append(res)
        print(
            f"    -> اصابت کامل! زمان: {res.intercept_time:.2f}s | "
            f"فاصله کمینه (Miss Distance): {res.miss_distance:.4f} m | "
            f"سرعت نهایی: {res.final_speed:.1f} m/s (ماخ {res.final_mach:.2f})"
        )

    # 5. Parametric trajectory fitting (Order >= 5)
    fitter = TrajectoryFitter(degree=poly_degree)
    fits: List[PolynomialFitResult] = []
    for res in results:
        fit = fitter.fit(res.time, res.r_M)
        fits.append(fit)

    # 6. Console Report & Parametric Equations
    display_results_table(results, fits)
    print_parametric_equations(results, fits)

    # 7. Visualizations
    viz = SimulationVisualizer()

    # Paths for saved figures
    suffix = f"mode{mode}_scen{scenario_id.upper()}"
    path_3d_png = os.path.join(output_dir, f"trajectory_3d_{suffix}.png")
    path_2d_png = os.path.join(output_dir, f"telemetry_2d_{suffix}.png")
    path_html = os.path.join(output_dir, f"intercept_3d_{suffix}.html")

    print("\n[*] در حال تولید پلات سه‌بعدی Matplotlib و گراف‌های تحلیلی زمانی ...")
    viz.plot_trajectories_3d_matplotlib(
        results,
        title="3D Missile vs Maneuvering Target Interception",
        num_los_rays=7,
        save_path=path_3d_png,
        show=show_plots,
    )

    viz.plot_telemetry_2d(
        results,
        title="Flight Dynamics & Guidance Telemetry",
        save_path=path_2d_png,
        show=show_plots,
    )

    print("[*] در حال تولید داشبورد سه‌بعدی وب تعاملی Plotly HTML ...")
    html_file = viz.generate_interactive_plotly_3d(
        results,
        title="Interactive 3D Interception Engagement (TPN vs APN vs Pure Pursuit)",
        output_html=path_html,
        num_los_rays=8,
    )

    print(f"\n[+] شبیه‌سازی با موفقیت کامل به پایان رسید.")
    print(f"    - نمودار سه‌بعدی ذخیره شد در: {os.path.abspath(path_3d_png)}")
    print(f"    - گراف‌های تحلیلی دوبعدی ذخیره شد در: {os.path.abspath(path_2d_png)}")
    print(f"    - داشبورد تعاملی ۳ بعدی ذخیره شد در: {os.path.abspath(html_file)}")

    return results, fits


def interactive_menu():
    """منوی تعاملی کنسول برای انتخاب آسان حالت‌ها و سناریوها توسط کاربر."""
    print("\n" + "=" * 65)
    print("   منوی تعاملی انتخاب سناریو و الگوریتم‌های شبیه‌سازی سه‌بعدی (3D)")
    print("=" * 65)
    print(" [1] حالت ۱: شبیه‌سازی پرواز موشک با الگوریتم 3D TPN")
    print(" [2] حالت ۲: شبیه‌سازی پرواز موشک با الگوریتم 3D APN")
    print(" [3] حالت ۳: شبیه‌سازی پرواز موشک با الگوریتم 3D Pure Pursuit (تعقیب محض - سر به هدف)")
    print(" [4] حالت ۴: مقایسه همزمان هر ۳ الگوریتم (TPN vs APN vs Pure Pursuit) [پیش‌فرض]")

    mode_input = input("\nلطفاً شماره حالت را وارد کنید (1-4) [پیش‌فرض 4]: ").strip()
    mode = int(mode_input) if mode_input in ["1", "2", "3", "4"] else 4

    print("\nانتخاب سناریوی مانور یا معادله ریاضی هدف:")
    print(" [A] سناریوی الف: پرواز یکنواخت مستقیم (Non-maneuvering Straight Line)")
    print(" [B] سناریوی ب: مانور مارپیچ/بشکه سه‌بعدی (3D Barrel Roll / Spiral Climb)")
    print(" [C] سناریوی ج: گردش با شتاب جانبی شدید در صفحه مایل (High-G Inclined Turn)")
    print(" [D] سناریوی د: مانور گریز زیگزاگی شدید (High-G 3D S-Turn / Break Turns) [پیشنهادی]")
    print(" [E / LINE] معادله خط سه‌بعدی هدف (Linear Vector Trajectory: r(t) = r0 + v*t)")
    print(" [F / PARABOLA] معادله سهمی سه‌بعدی هدف (Parabolic Path: r(t) = r0 + v0*t + 0.5*a*t²)")
    print(" [G / CUSTOM] معادله توابع دلخواه ریاضی سه‌بعدی (Custom: x(t), y(t), z(t))")

    scen_input = input("\nلطفاً شناسه سناریو را وارد کنید [پیش‌فرض D]: ").strip().upper()
    target_kwargs = {}

    if scen_input in ["E", "LINE"]:
        scenario = "LINE"
        vx_in = input("مؤلفه جهت vx [پیش‌فرض -250.0]: ").strip()
        vx = float(vx_in) if vx_in else -250.0
        vy_in = input("مؤلفه جهت vy [پیش‌فرض -50.0]: ").strip()
        vy = float(vy_in) if vy_in else -50.0
        vz_in = input("مؤلفه جهت vz [پیش‌فرض 20.0]: ").strip()
        vz = float(vz_in) if vz_in else 20.0
        target_kwargs = {"heading_dir": np.array([vx, vy, vz], dtype=np.float64)}
    elif scen_input in ["F", "PARABOLA"]:
        scenario = "PARABOLA"
        ax_in = input("شتاب ax [پیش‌فرض 0.0]: ").strip()
        ax = float(ax_in) if ax_in else 0.0
        ay_in = input("شتاب ay [پیش‌فرض 15.0]: ").strip()
        ay = float(ay_in) if ay_in else 15.0
        az_in = input("شتاب az [پیش‌فرض -9.81]: ").strip()
        az = float(az_in) if az_in else -9.81
        target_kwargs = {"a_const": np.array([ax, ay, az], dtype=np.float64)}
    elif scen_input in ["G", "CUSTOM"]:
        scenario = "CUSTOM"
        x_in = input("معادله x(t) [پیش‌فرض 6000 - 240*t]: ").strip()
        y_in = input("معادله y(t) [پیش‌فرض 2500 + 350*sin(0.5*t)]: ").strip()
        z_in = input("معادله z(t) [پیش‌فرض 5000 + 200*cos(0.5*t)]: ").strip()
        x_expr = x_in if x_in else "6000 - 240*t"
        y_expr = y_in if y_in else "2500 + 350*sin(0.5*t)"
        z_expr = z_in if z_in else "5000 + 200*cos(0.5*t)"
        target_kwargs = {"x_expr": x_expr, "y_expr": y_expr, "z_expr": z_expr}
    else:
        scenario = scen_input if scen_input in ["A", "B", "C", "D"] else "D"

    return mode, scenario, target_kwargs


def main():
    parser = argparse.ArgumentParser(
        description="3D Industrial Physical Simulation of Missile-Target Interception (TPN vs APN vs Pure Pursuit)"
    )
    parser.add_argument(
        "--mode",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Execution Mode: 1 (TPN only), 2 (APN only), 3 (Pure Pursuit), 4 (Side-by-Side 3-Way comparison)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target Maneuver Scenario: A, B, C, D, LINE, PARABOLA, or CUSTOM",
    )
    parser.add_argument("--custom-x", type=str, default="6000 - 240*t")
    parser.add_argument("--custom-y", type=str, default="2500 + 350*sin(0.5*t)")
    parser.add_argument("--custom-z", type=str, default="5000 + 200*cos(0.5*t)")
    parser.add_argument("--line-vx", type=float, default=-250.0)
    parser.add_argument("--line-vy", type=float, default=-50.0)
    parser.add_argument("--line-vz", type=float, default=20.0)
    parser.add_argument("--line-speed", type=float, default=300.0)
    parser.add_argument("--parabola-ax", type=float, default=0.0)
    parser.add_argument("--parabola-ay", type=float, default=15.0)
    parser.add_argument("--parabola-az", type=float, default=-9.81)
    parser.add_argument(
        "--poly-deg",
        type=int,
        default=6,
        help="Polynomial fitting degree (minimum 5, default 6)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display interactive Matplotlib GUI windows (useful for headless / automated runs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "outputs"),
        help="Directory to save generated plots and HTML dashboard",
    )

    args = parser.parse_args()

    # Print theoretical documentation banner
    print_theoretical_and_physical_brief()

    target_kwargs = {}
    if args.mode is None or args.target is None:
        if sys.stdin.isatty():
            mode, scenario, target_kwargs = interactive_menu()
        else:
            mode = 4
            scenario = "D"
            print(f"[Auto-Select] Running default Mode {mode} (3-Way Comparison) with Scenario {scenario} (High-G 3D S-Turn).")
    else:
        mode = args.mode
        scenario = args.target.upper().strip()
        if scenario == "LINE":
            target_kwargs = {
                "heading_dir": np.array([args.line_vx, args.line_vy, args.line_vz], dtype=np.float64),
                "speed": args.line_speed,
            }
        elif scenario == "PARABOLA":
            target_kwargs = {
                "a_const": np.array([args.parabola_ax, args.parabola_ay, args.parabola_az], dtype=np.float64),
            }
        elif scenario == "CUSTOM":
            target_kwargs = {
                "x_expr": args.custom_x,
                "y_expr": args.custom_y,
                "z_expr": args.custom_z,
            }

    run_pipeline(
        mode=mode,
        scenario_id=scenario,
        target_kwargs=target_kwargs,
        poly_degree=args.poly_deg,
        show_plots=(not args.no_show) and sys.stdin.isatty(),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
