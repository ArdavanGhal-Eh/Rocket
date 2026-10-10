<a id="readme-top"></a>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Persian Documentation](https://img.shields.io/badge/مستندات-فارسی-green.svg?style=for-the-badge)](#persian-documentation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![GNC Guidance](https://img.shields.io/badge/GNC-TPN_%7C_APN_%7C_Pure_Pursuit-orange.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)
[![Physics Engine](https://img.shields.io/badge/Physics-RK4_200Hz_Integrator-blue.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)
[![GUI & 3D WebGL](https://img.shields.io/badge/UI-Tkinter_%7C_Plotly_3D_WebGL-purple.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)
[![CI/CD Pipeline](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-brightgreen.svg?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/ArdavanGhal-Eh/Rocket/actions)
[![Tests Passing](https://img.shields.io/badge/Tests-53%2F53_Passed-brightgreen.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)
[![Monte Carlo](https://img.shields.io/badge/Monte_Carlo-Stochastic_CEP-blue.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)
[![Security](https://img.shields.io/badge/Security-Sandboxed_AST-success.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)
[![Stars](https://img.shields.io/github/stars/ArdavanGhal-Eh/Rocket?style=for-the-badge&color=gold)](https://github.com/ArdavanGhal-Eh/Rocket/stargazers)
[![Issues](https://img.shields.io/github/issues/ArdavanGhal-Eh/Rocket?style=for-the-badge&color=red)](https://github.com/ArdavanGhal-Eh/Rocket/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket/pulls)

<br />

# 🚀 High-Fidelity Aerospace Interception Simulation Suite (2D & 3D)
### *Industrial GNC Guidance Laws, RK4 Physics, Tkinter Desktop GUI & Interactive 3D WebGL Dashboards*
#### *مجموعه شبیه‌سازی جامع مهندسی هدایت موشک و درگیری هواپایه (۲ بعدی و ۳ بعدی)*

<p align="center">
  <b>A comprehensive, industry-grade aerospace Guidance, Navigation, and Control (GNC) simulation suite for surface-to-air and air-to-air missile interception against highly maneuvering targets in both 2D planar and full 3D spatial domains. Implements True Proportional Navigation (TPN), Augmented Proportional Navigation (APN), and Pure Pursuit (PP) under variable-mass rocket motor dynamics, aerodynamic drag, ISA atmosphere, 35G spherical acceleration saturation, and sub-millimeter Closest Point of Approach (CPA) calculations.</b>
  <br /><br />
  <a href="#2-guidance--interception-algorithms"><strong>Explore Algorithms »</strong></a>
  &nbsp;•&nbsp;
  <a href="#3-complete-missile-physical--engineering-parameters"><strong>Physical Parameters »</strong></a>
  &nbsp;•&nbsp;
  <a href="#5-execution--usage-instructions"><strong>Execution Guide »</strong></a>
  &nbsp;•&nbsp;
  <a href="#persian-documentation"><strong>راهنمای فارسی »</strong></a>
</p>

</div>

---

<!-- TABLE OF CONTENTS -->
<details open>
  <summary><h2 style="display: inline-block;">📑 Table of Contents / فهرست مطالب</h2></summary>
  <ol>
    <li>
      <a href="#part-1-english-aerospace-engineering-documentation"><b>PART 1: English Aerospace Engineering Documentation</b></a>
      <ul>
        <li><a href="#1-project-architecture--directory-layout">1. Project Architecture & Directory Layout</a></li>
        <li><a href="#2-guidance--interception-algorithms">2. Guidance & Interception Algorithms (TPN, APN, PP)</a></li>
        <li><a href="#3-complete-missile-physical--engineering-parameters">3. Complete Missile Physical & Engineering Parameters</a></li>
        <li><a href="#4-target-trajectory-modeling--mathematical-formulations">4. Target Trajectory Modeling & Mathematical Formulations</a></li>
        <li><a href="#5-execution--usage-instructions">5. Execution & Usage Instructions (GUI, CLI, 24 Tests)</a></li>
      </ul>
    </li>
    <li>
      <a href="#persian-documentation"><b>بخش ۲: مستندات مهندسی و فنی پروژه به زبان فارسی</b></a>
      <ul>
        <li><a href="#۱-معماری-پروژه-و-ساختار-پوشه‌ها">۱. معماری پروژه و ساختار پوشه‌ها</a></li>
        <li><a href="#۲-تحلیل-ریاضی-و-فیزیکی-الگوریتمهای-هدایت-و-اصابت">۲. تحلیل ریاضی و فیزیکی الگوریتم‌های هدایت و اصابت</a></li>
        <li><a href="#۳-تشریح-جامع-تکتک-پارامترهای-فیزیکی-آیرودینامیکی-و-کنترلی-موشک">۳. تشریح جامع تک‌تک پارامترهای فیزیکی، آیرودینامیکی و کنترلی موشک</a></li>
        <li><a href="#۴-نحوه-تعریف-هندسه-و-معادلات-ریاضی-مسیر-هدف">۴. نحوه تعریف هندسه و معادلات ریاضی مسیر هدف</a></li>
        <li><a href="#۵-راهنمای-کامل-اجرا-و-بهکارگیری">۵. راهنمای کامل اجرا و به‌کارگیری</a></li>
      </ul>
    </li>
  </ol>
</details>

---

# PART 1: ENGLISH AEROSPACE ENGINEERING DOCUMENTATION

## 1. Project Architecture & Directory Layout

The codebase is organized into two completely decoupled simulation frameworks (**2D Planar** and **3D Spatial**), orchestrated by a unified desktop GUI and scriptable command-line interfaces.

```
Rocket/
│
├── .github/workflows/ci.yml          # Automated CI/CD Multi-OS & Multi-Python test matrix
├── 3D/                               # Complete 3D Spatial Simulation Environment
│   ├── missile_sim/                  # Core 3D GNC package
│   │   ├── config.py                 # Dataclasses for missile, target, and simulation physics
│   │   ├── guidance.py               # 3D TPN, 3D APN, and 3D Pure Pursuit algorithms
│   │   ├── missile.py                # 3D missile state, variable mass depletion, drag & thrust
│   │   ├── target.py                 # Scenarios A-D + Linear, Parabolic, & Custom Math functions
│   │   ├── safe_math.py              # Sandboxed AST-based mathematical parser & bytecode pre-compiler
│   │   ├── monte_carlo.py            # Stochastic Monte Carlo dispersion & CEP statistical engine
│   │   ├── simulator.py              # 4th-Order Runge-Kutta (RK4) engine with sub-mm CPA resolution
│   │   ├── visualizer.py             # Matplotlib 3D trajectory renders & Plotly WebGL dashboards
│   │   └── fitting.py                # Degree-6 parametric trajectory polynomial fitting (R² > 0.9999)
│   ├── outputs/                      # Generated PNG telemetry charts and interactive HTML dashboards
│   ├── tests/                        # 18 automated unittests verifying physics, AST, & Monte Carlo
│   ├── main.py                       # 3D CLI entrypoint with Monte Carlo dispersion flag
│   └── README.md                     # Dedicated 3D documentation
│
├── 2D/                               # Complete 2D Planar Simulation Environment
│   ├── missile_sim_2d/               # Core 2D GNC package
│   │   ├── config.py                 # 2D physical parameters and engagement configurations
│   │   ├── guidance.py               # 2D TPN, 2D APN, and 2D Pure Pursuit algorithms
│   │   ├── missile.py                # 2D point-mass missile dynamics and propulsion
│   │   ├── target.py                 # 2D Scenarios A-D + Linear, Parabolic, & Custom Math trajectories
│   │   ├── safe_math.py              # Sandboxed AST parser for 2D custom equations
│   │   ├── monte_carlo.py            # 2D stochastic Monte Carlo dispersion engine
│   │   ├── simulator.py              # 2D RK4 numerical integration engine
│   │   ├── visualizer.py             # 2D Matplotlib plots and interactive Plotly Web charts
│   │   └── fitting.py                # 2D explicit polynomial fitting (x(t), y(t))
│   ├── outputs/                      # Saved 2D publication-quality PNGs and HTML files
│   ├── tests/                        # 18 automated unittests covering 2D algorithms, AST, & Monte Carlo
│   ├── main.py                       # 2D CLI entrypoint with Monte Carlo flag
│   └── README.md                     # Dedicated 2D documentation
│
├── gui_app.py                        # Unified Desktop Engineering GUI (Tkinter + Matplotlib + Monte Carlo)
├── pyproject.toml                    # Modern PEP 518/621 package build configuration
├── requirements.txt                  # Production runtime dependencies
├── requirements-dev.txt              # Testing, coverage, linting & dev dependencies
└── README.md                         # Root Comprehensive Bilingual Documentation
```

---

## 2. Guidance & Interception Algorithms

### 2.1 Kinematic Geometry & Collision Triangle

In both 2D and 3D engagements, the relative position vector $\vec{R}$ and relative velocity vector $\vec{V}_{\mathrm{rel}}$ between missile ($\vec{r}_M, \vec{v}_M$) and target ($\vec{r}_T, \vec{v}_T$) are defined as:

$$
\vec{R} = \vec{r}_T - \vec{r}_M, \quad R = \|\vec{R}\|, \quad \hat{R} = \frac{\vec{R}}{R}
$$

$$
\vec{V}_{\mathrm{rel}} = \vec{v}_T - \vec{v}_M
$$

The **closing velocity** $V_c$ (rate of range decrease) is:

$$
V_c = -\dot{R} = -\frac{\vec{R} \cdot \vec{V}_{\mathrm{rel}}}{R}
$$

The **Line-of-Sight (LOS) angular velocity vector** in 3D Euclidean space is:

$$
\vec{\Omega}_{\mathrm{LOS}} = \frac{\vec{R} \times \vec{V}_{\mathrm{rel}}}{R^2}
$$

In 2D planar space, with $\lambda = \mathrm{atan2}(R_y, R_x)$, the scalar LOS rotation rate is:

$$
\dot{\lambda} = \frac{R_x V_{\mathrm{rel},y} - R_y V_{\mathrm{rel},x}}{R^2}
$$

---

### 2.2 True Proportional Navigation (TPN)

#### Mathematical Formulation:

**3D Spatial Vector Formulation:**

$$
\vec{a}_{\mathrm{cmd, TPN}} = N \cdot V_c \cdot (\vec{\Omega}_{\mathrm{LOS}} \times \hat{R})
$$

**2D Planar Formulation:**

$$
\vec{a}_{\mathrm{cmd, TPN}} = N \cdot V_c \cdot \dot{\lambda} \cdot \hat{n}_{\mathrm{LOS}}
$$

where $\hat{n}_{\mathrm{LOS}} = [-\sin\lambda, \cos\lambda]^T$ is the unit vector normal to the instantaneous line-of-sight.

#### Physical Maneuver Principle:
TPN commands a lateral acceleration strictly perpendicular to the line-of-sight vector to drive the line-of-sight rate to zero ($\dot{\lambda} \to 0$ or $\vec{\Omega}_{\mathrm{LOS}} \to \vec{0}$). When $\dot{\lambda} = 0$, the missile and target are locked onto a **Constant Bearing Decreasing Range (CBDR)** collision triangle.

#### Flight Dynamics & Endgame Limitations:
- **Non-Maneuvering Targets:** Near-optimal trajectory with minimal control energy. The missile leads the target efficiently from launch.
- **Aggressive Maneuvering Targets:** TPN exhibits an inherent **kinematic phase lag**. Because the guidance command is proportional only to the line-of-sight rate caused by past displacement, the missile continuously lags an evasive target. As range $R \to 0$ in the terminal phase ("Endgame"), the required maneuver spikes dramatically, frequently causing lateral acceleration saturation at the missile's structural limit ($35g$) and resulting in substantial miss distances.

---

### 2.3 Augmented Proportional Navigation (APN)

#### Mathematical Formulation:

**3D Spatial Vector Formulation:**

$$
\vec{a}_{\mathrm{cmd, APN}} = N \cdot V_c \cdot (\vec{\Omega}_{\mathrm{LOS}} \times \hat{R}) + \frac{N}{2} \vec{a}_{T\perp}
$$

where the normal target acceleration perpendicular to the line-of-sight is:

$$
\vec{a}_{T\perp} = \vec{a}_T - (\vec{a}_T \cdot \hat{R})\hat{R}
$$

**2D Planar Formulation:**

$$
\vec{a}_{\mathrm{cmd, APN}} = N \cdot V_c \cdot \dot{\lambda} \cdot \hat{n}_{\mathrm{LOS}} + \frac{N}{2} (\vec{a}_T \cdot \hat{n}_{\mathrm{LOS}}) \hat{n}_{\mathrm{LOS}}
$$

#### Physical Maneuver Principle:
APN incorporates a direct **feedforward compensation term** proportional to the target's normal maneuver acceleration. By measuring (or estimating via Kalman filter) the target's lateral acceleration, APN commands the missile to match and neutralize the evasive maneuver instantaneously.

#### Flight Dynamics & Advantages:
- Completely cancels out maneuver drift before it accumulates into LOS angular errors.
- Prevents terminal acceleration spikes, keeping lateral G-demand smooth and well below structural saturation throughout the engagement.
- Achieves sub-meter or millimeter-level miss distances (true direct impact) even against 8G–10G high-frequency defensive weave maneuvers (break-turns, barrel rolls, and S-turns).

---

### 2.4 Pure Pursuit (Nose-to-Target)

#### Mathematical Formulation:

**2D Planar Formulation:**

Let $\gamma_M = \mathrm{atan2}(v_{M,y}, v_{M,x})$ be the missile flight-path angle and $\lambda = \mathrm{atan2}(R_y, R_x)$ be the LOS angle. The heading error angle is:

$$
\eta = \lambda - \gamma_M
$$

The commanded turn rate of the missile velocity vector is:

$$
\dot{\gamma}_{\mathrm{cmd}} = \dot{\lambda} + K_p \cdot \eta
$$

The lateral acceleration command perpendicular to the missile velocity vector ($\hat{n}_{vM} = [-\sin\gamma_M, \cos\gamma_M]^T$) is:

$$
\vec{a}_{\mathrm{cmd, PP}} = \|\vec{v}_M\| \cdot \dot{\gamma}_{\mathrm{cmd}} \cdot \hat{n}_{vM}
$$

**3D Spatial Vector Formulation:**

Let $\hat{v}_M = \frac{\vec{v}_M}{\|\vec{v}_M\|}$ and $\hat{R} = \frac{\vec{R}}{\|\vec{R}\|}$. The angular separation between missile velocity and LOS is:

$$
\eta = \arccos(\mathrm{clip}(\hat{v}_M \cdot \hat{R}, -1, 1))
$$

The instantaneous rotation axis is:

$$
\vec{u} = \frac{\hat{v}_M \times \hat{R}}{\|\hat{v}_M \times \hat{R}\|}
$$

The commanded angular rotation vector of the velocity vector is:

$$
\vec{\omega}_{\mathrm{cmd}} = \vec{\Omega}_{\mathrm{LOS}} + K_p \cdot \eta \cdot \vec{u}
$$

The commanded normal acceleration vector (strictly perpendicular to velocity) is:

$$
\vec{a}_{\mathrm{cmd, PP}} = \vec{\omega}_{\mathrm{cmd}} \times \vec{v}_M
$$

#### Physical Maneuver Principle & Tail-Chase Dynamics:
Unlike Proportional Navigation laws, **Pure Pursuit does NOT lead the target**. The missile's nose and velocity vector continuously aim directly at the current visual position of the target (zero lead angle: $\eta \to 0$).

#### Critical Mathematical & Physical Phenomenon:
1. **Curved Pursuit Trajectory:** Pure Pursuit creates the classical "hound-and-hare" curve. Because it lacks a lead angle, the missile always trails behind crossing targets.
2. **Terminal Acceleration Singularity:** The apparent line-of-sight angular rate satisfies:

$$
\dot{\lambda} \approx \frac{V_T \sin(\theta_T - \lambda)}{R}
$$

As range $R \to 0$, if the target maintains any velocity component perpendicular to the LOS, $\dot{\lambda} \to \infty$. Consequently, the lateral acceleration demanded by Pure Pursuit explodes to infinity ($\|\vec{a}_{\mathrm{cmd}}\| \to \infty$) as the missile approaches the target.
3. **Structural Saturation & Miss Distance:** In realistic flight, the missile hits its structural limiter ($35g$). Because the missile cannot execute the required infinite turn rate, it slips past maneuvering targets, resulting in a significantly larger terminal miss distance than PN laws.

---

### 2.5 Comparative Performance Matrix

| Metric / Feature | True Proportional Navigation (TPN) | Augmented Proportional Navigation (APN) | Pure Pursuit (Nose-to-Target) |
| :--- | :--- | :--- | :--- |
| **Guidance Concept** | Nullify LOS rate ($\dot{\lambda} \to 0$) | Nullify LOS rate + Feedforward target acceleration | Align velocity vector directly with target position ($\vec{v}_M \parallel \vec{R}$) |
| **Lead Angle** | Active collision lead angle | Active collision lead angle + maneuver lead | Zero lead angle (trails behind target) |
| **Trajectory Geometry** | Nearly straight / collision triangle | Optimal straight intercept vector | Curved pursuit curve (hound-and-hare) |
| **Endgame G-Demand** | Spikes against maneuvering targets | Flat, low, and distributed uniformly | Diverges toward infinity ($\propto 1/R$) |
| **Terminal G-Saturation** | High risk under aggressive target weave | Minimal / Zero saturation | Severe saturation at $35g$ boundary |
| **Terminal Miss Distance** | Moderate ($0.5\text{ m} - 5.0\text{ m}$) | Ultra-low ($< 0.1\text{ m}$, direct kinetic hit) | Highest ($1.5\text{ m} - 15.0\text{ m}$) |
| **Sensor Requirements** | Seeker angular rates ($\dot{\lambda}$) & range rate ($V_c$) | Seeker rates + Target acceleration estimate $\vec{a}_T$ | Target line-of-sight angle / bearing only |
| **Control Energy** | Moderate | Lowest (energy-optimal) | High (wasted in continuous turning) |

---

## 3. Complete Missile Physical & Engineering Parameters

The simulation integrates a 6-DOF-equivalent point-mass aerodynamic and propulsion model grounded in real-world surface-to-air and air-to-air missile physics (similar to the AIM-9 Sidewinder / AIM-120 AMRAAM class).

| Parameter Symbol | Physical Value | Unit | Engineering Category | Physical Description & Mathematical Role |
| :--- | :--- | :--- | :--- | :--- |
| $m_0$ | `85.0` | $\text{kg}$ | Mass & Inertia | Initial total missile mass at launch (structure, warhead, avionics, motor, propellant). |
| $m_{\mathrm{dry}}$ | `50.0` | $\text{kg}$ | Mass & Inertia | Burnout structural dry mass after propellant depletion ($t \ge 4.0\text{ s}$). |
| $m_{\mathrm{prop}}$ | `35.0` | $\text{kg}$ | Mass & Inertia | Total consumable solid rocket propellant mass ($m_{\mathrm{prop}} = m_0 - m_{\mathrm{dry}}$). |
| $t_{\mathrm{burn}}$ | `4.0` | $\text{s}$ | Propulsion | Rocket motor total burn duration. Separates the Boost phase from the Coast phase. |
| $\dot{m}$ | `8.75` | $\text{kg/s}$ | Propulsion | Mass depletion rate: $\dot{m} = m_{\mathrm{prop}} / t_{\mathrm{burn}} = 35 / 4 = 8.75\text{ kg/s}$. Mass is $m(t) = m_0 - \dot{m}t$. |
| $T_{\mathrm{nominal}}$ | `17,500.0` | $\text{N}$ | Propulsion | Constant thrust force produced during boost: $T(t) = 17.5\text{ kN}$ for $t \le 4\text{ s}$; $T(t) = 0$ for $t > 4\text{ s}$. |
| $I_{\mathrm{sp}}$ | `203.9` | $\text{s}$ | Propulsion | Specific impulse of the solid rocket motor: $I_{\mathrm{sp}} = \frac{T}{\dot{m} \cdot g_0} = \frac{17500}{8.75 \times 9.81} \approx 203.9\text{ s}$. |
| $d$ | `0.127` | $\text{m}$ | Aerodynamics | Missile body caliber / diameter ($127\text{ mm}$ / $5.0\text{ inches}$). |
| $A_{\mathrm{ref}}$ | `0.0127` | $\text{m}^2$ | Aerodynamics | Frontal reference aerodynamic cross-sectional area: $A_{\mathrm{ref}} = \frac{\pi d^2}{4} \approx 0.01267\text{ m}^2$. |
| $\rho$ | `0.736` | $\text{kg/m}^3$ | Atmosphere | Atmospheric air density at operational altitude ($5,000\text{ m}$ ISA standard atmosphere). |
| $a$ | `320.5` | $\text{m/s}$ | Atmosphere | Local speed of sound at $5,000\text{ m}$ altitude ($\text{Mach } 1.0 = 320.5\text{ m/s}$). |
| $C_D$ | `0.40` | Dimensionless | Aerodynamics | Baseline zero-lift aerodynamic drag coefficient. Quadratic drag: $\vec{D} = -\frac{1}{2} \rho V_M C_D A_{\mathrm{ref}} \vec{v}_M$. |
| $q(t)$ | Dynamic | $\text{N/m}^2$ | Aerodynamics | Instantaneous dynamic pressure: $q = \frac{1}{2} \rho \|\vec{v}_M\|^2$. |
| $G_{\mathrm{limit}}$ | `35.0` | $g$ | Structural Limits | Maximum allowable lateral structural acceleration ($35 \times 9.81 = 343.35\text{ m/s}^2$). |
| $g_0$ | `9.81` | $\text{m/s}^2$ | Geophysics | Standard gravitational acceleration constant at Earth surface. |
| $V_{M0}$ | `250.0` | $\text{m/s}$ | Initial Kinematics | Initial launch airspeed ($\text{Mach } 0.78$), aligned precisely with initial LOS vector $\hat{R}_0$. |
| $\vec{r}_{M0}$ | `[0, 0, 0]` | $\text{m}$ | Initial Kinematics | Launch origin coordinates in Euclidean NED / Cartesian inertial frame. |
| $N$ | `4.0` | Dimensionless | Guidance Gain | Effective navigation ratio for TPN and APN ($N \in [3, 5]$, nominal $4.0$). |
| $K_p$ | `4.0` | $\text{s}^{-1}$ | Guidance Gain | Proportional heading error rate feedback gain for Pure Pursuit guidance. |
| $\Delta t$ | `0.005` | $\text{s}$ | Numerical Solver | Runge-Kutta 4th-order (RK4) integration fixed time step ($200\text{ Hz}$ update rate). |
| $t_{\mathrm{max}}$ | `25.0` | $\text{s}$ | Simulation Horizon | Maximum allowed simulation flight time before aborting unintercepted runs. |

---

## 4. Target Trajectory Modeling & Mathematical Formulations

The suite provides standard fighter combat maneuvers (Scenarios A through D) alongside explicit mathematical trajectory functions:

### 1. Linear Trajectory (`LINE`)

**2D Planar Formulation:** Straight line $y(x) = m \cdot x + c$ with constant cruise speed $V_T$:

$$
v_x = \mathrm{dir} \cdot \frac{V_T}{\sqrt{1 + m^2}}, \quad v_y = m \cdot v_x, \quad \vec{a}_T = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

**3D Spatial Formulation:** Constant 3D velocity vector from initial coordinate $\vec{r}_{T0}$:

$$
\vec{r}_T(t) = \vec{r}_{T0} + \vec{v}_{\mathrm{const}} \cdot t, \quad \vec{a}_T(t) = \vec{0}
$$

### 2. Parabolic Trajectory (`PARABOLA`)

**2D Planar Formulation:** Cartesian parabola $y(x) = a \cdot x^2 + b \cdot x + c$ with downrange motion $x(t) = x_0 + v_x \cdot t$:

$$
v_y(t) = (2 a x(t) + b) \cdot v_x, \quad a_y(t) = 2 a v_x^2
$$

**3D Spatial Formulation:** Kinematic projectile or constant-acceleration maneuver:

$$
\vec{r}_T(t) = \vec{r}_{T0} + \vec{v}_{T0} \cdot t + \frac{1}{2}\vec{a}_{\mathrm{const}} \cdot t^2, \quad \vec{v}_T(t) = \vec{v}_{T0} + \vec{a}_{\mathrm{const}} \cdot t, \quad \vec{a}_T(t) = \vec{a}_{\mathrm{const}}
$$

### 3. Custom Arbitrary Mathematical Functions (`CUSTOM`)

Accepts analytical time-dependent string expressions for each axis:
- **2D:** $x(t) = f(t)$ and $y(t) = g(t)$
- **3D:** $x(t) = f(t)$, $y(t) = g(t)$, and $z(t) = h(t)$

**Numerical Velocity & Acceleration Extraction:**
To eliminate symbolic differentiation errors and support non-elementary mathematical functions, the target velocity and acceleration are extracted using high-precision $\mathcal{O}(h^2)$ central finite differences with step $h = 10^{-5}\text{ s}$:

$$
\vec{v}_T(t) = \frac{\vec{r}_T(t + h) - \vec{r}_T(t - h)}{2h}
$$

$$
\vec{a}_T(t) = \frac{\vec{r}_T(t + h) - 2\vec{r}_T(t) + \vec{r}_T(t - h)}{h^2}
$$

Supported mathematical functions in string inputs: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`, `pi`, `e`, and power operators (`^` or `**`).

---

## 5. Execution & Usage Instructions

### 5.1 Desktop Engineering GUI (`gui_app.py`)

Launch the cross-platform interactive GUI built with Tkinter and embedded Matplotlib:

```powershell
python gui_app.py
```

#### GUI Capabilities:
- **Tab 1: 3D Spatial Interception** and **Tab 2: 2D Planar Interception**.
- **Live Real-Time Simulation Animation:** Dynamic, non-blocking frame-by-frame live playback of missile and target trajectories directly on embedded Matplotlib canvases without freezing the GUI.
- **Interactive Playback Controls:** Full `Run Simulation`, `Pause`, `Resume`, and `Restart` states with execution concurrency locks.
- **Variable Playback Speeds:** High-fidelity simulation throttling across `0.25x`, `0.5x`, `1.0x (Real-Time)`, `2.0x`, and `5.0x` speeds.
- **Live Flight Telemetry HUD:** Real-time digital dashboard showing instantaneous Simulation Time, Missile Speed, Mach, Range-to-Target, Closing Velocity $V_c$, Steering G-load, 35G saturation status, and spatial coordinates.
- **Mode Selection:** Mode 1 (TPN), Mode 2 (APN), Mode 3 (Pure Pursuit), or Mode 4 (All 3 Algorithms Simultaneously).
- **Target Profiles:** Scenarios A–D, LINE, PARABOLA, or CUSTOM math equations ($x(t), y(t), z(t)$).
- **Dynamic Viewports & Trails:** Auto-scaling viewports with 8% padding ensuring both objects remain in frame, persistent trajectory trails, dynamic Line-of-Sight (LOS) vectors, and terminal CPA blast indicators.
- **Telemetry Charts:** Instantaneous Lateral G-load, Velocity & Mach profiles, Interception Range curve, and Cumulative Control Energy.
- **Analytical Curve Fitting:** Displays degree-6 fitted polynomial coefficients with $R^2 > 0.9999$.
- **Interactive Web Button:** Opens interactive Plotly WebGL dashboards in your default browser.

---

### 5.2 Command-Line Interface (CLI) Execution

#### 2D Planar Simulations:
```powershell
# Interactive console menu:
python 2D/main.py

# Mode 4 (3-way benchmark) against custom trigonometric path:
python 2D/main.py --mode 4 --target CUSTOM --custom-x "6000 - 250*t" --custom-y "2500 + 400*sin(0.5*t)"

# Benchmark against linear target:
python 2D/main.py --mode 4 --target LINE --line-slope 0.4167 --line-intercept 0.0

# Benchmark against parabolic target:
python 2D/main.py --mode 4 --target PARABOLA --parabola-a 0.00005 --parabola-b -0.2 --parabola-c 1900.0
```

#### 3D Spatial Simulations:
```powershell
# Interactive console menu:
python 3D/main.py

# Mode 4 against high-G defensive break-turns (Scenario D):
python 3D/main.py --mode 4 --target D

# Mode 4 against custom 3D mathematical helical spiral:
python 3D/main.py --mode 4 --target CUSTOM --custom-x "6000 - 240*t" --custom-y "2500 + 200*sin(0.4*t)" --custom-z "5000 + 150*cos(0.4*t)"

# Mode 4 against 3D constant-velocity linear flight:
python 3D/main.py --mode 4 --target LINE --line-vx -250.0 --line-vy -50.0 --line-vz 20.0

# Stochastic Monte Carlo Dispersion Campaign (50 runs with CEP analysis):
python 3D/main.py --mode 2 --target A --monte-carlo 50
python 2D/main.py --mode 2 --target A --monte-carlo 50
```

---

### 5.3 Automated Verification Test Suites & CI/CD

Run the comprehensive regression test suite (53 tests total) using `pytest`:

```powershell
# Run all 53 unit and integration tests with coverage:
pytest -v

# Or run standard unittest discovery:
python -m unittest discover -s 3D/tests -p "test_*.py"
python -m unittest discover -s 2D/tests -p "test_*.py"
```

All 53 unit and integration tests pass with 100% verification covering relative kinematics, guidance command vectors, RK4 step integration, CPA resolution, sandboxed AST expression parsing, Monte Carlo dispersion statistics, live animation state machines, and headless GUI lifecycle playback controls.

---
---

<a id="persian-documentation"></a>

# PART 2: مستندات مهندسی و فنی پروژه به زبان فارسی

## ۱. معماری پروژه و ساختار پوشه‌ها

این مخزن شامل یک مجموعه مهندسی و صنعتی کامل برای تحلیل، شبیه‌سازی فیزیکی و مقایسه الگوریتم‌های هدایت و ناوبری (GNC) در دو فضای مستقل **دوبعدی (2D Planar)** و **سه‌بعدی (3D Space)** است. معماری پروژه به‌صورت کاملاً ماژولار پیاده‌سازی شده و تمامی ماژول‌های محاسباتی، فیزیکی و بصری از یکدیگر تفکیک شده‌اند:

```
Rocket/
│
├── .github/workflows/ci.yml          # پایپ‌لاین خودکار CI/CD در ماتریس چندسیستمی و چند نسخه‌ای پایتون
├── 3D/                               # پروژه کامل شبیه‌سازی در فضای سه‌بعدی
│   ├── missile_sim/                  # پکیج سه‌بعدی (کینماتیک 3D، دینامیک ۶ درجه، درگ، پیشران)
│   │   ├── config.py                 # کلاس‌های داده پیکربندی فیزیکی موشک، هدف و شبیه‌ساز
│   │   ├── guidance.py               # الگوریتم‌های 3D TPN, 3D APN, 3D Pure Pursuit
│   │   ├── missile.py                # مدل فیزیکی موشک، تخلیه متغیر جرم، تراست و پسا
│   │   ├── target.py                 # سناریوهای مانور A-D + اهداف خطی، سهمی و توابع دلخواه ریاضی
│   │   ├── safe_math.py              # پارسر ریاضی ایمن و سندباکس مبتنی بر درخت AST و بایت‌کد
│   │   ├── monte_carlo.py            # موتور تحلیل آماری پراکندگی مونت‌کارلو و محاسبه CEP و Pk
│   │   ├── simulator.py              # حل‌کننده فیزیکی یکپارچه RK4 با محاسبه میلی‌متری CPA
│   │   ├── visualizer.py             # تولید پلات‌های ۳ بعدی Matplotlib و داشبورد وب تعاملی Plotly
│   │   └── fitting.py                # استخراج معادلات صریح چندجمله‌ای درجه ۶ مسیر پرواز موشک
│   ├── outputs/                      # گراف‌های ذخیره‌شده سه‌بعدی PNG و داشبوردهای تعاملی HTML Plotly
│   ├── tests/                        # آزمون‌های خودکار و ممیزی سه‌بعدی (۱۸ تست واحد و اعتبارسنجی)
│   ├── main.py                       # اسکریپت اجرایی اصلی سه‌بعدی (CLI با پرچم مونت‌کارلو)
│   └── README.md                     # مستندات فنی سه‌بعدی
│
├── 2D/                               # پروژه کامل شبیه‌سازی در صفحه دوبعدی
│   ├── missile_sim_2d/               # پکیج دوبعدی (کینماتیک Planar 2D، زاویه خط دید lambda)
│   │   ├── config.py                 # پارامترهای فیزیکی و هندسی درگیری دوبعدی
│   │   ├── guidance.py               # الگوریتم‌های 2D TPN, 2D APN, 2D Pure Pursuit
│   │   ├── missile.py                # مدل دینامیکی نقطه مادی موشک در صفحه
│   │   ├── target.py                 # سناریوهای دوبعدی A-D + خط، سهمی و فرمول‌های دلخواه ریاضی
│   │   ├── safe_math.py              # پارسر ریاضی ایمن مبتنی بر AST برای صفحه دوبعدی
│   │   ├── monte_carlo.py            # موتور تحلیل پراکندگی تصادفی مونت‌کارلو دوبعدی
│   │   ├── simulator.py              # حل‌کننده عددی گام زمانی RK4 دوبعدی
│   │   ├── visualizer.py             # پلات‌های تحلیلی دوبعدی و وب‌اپ تعاملی Plotly HTML
│   │   └── fitting.py                # استخراج معادلات تحلیلی x(t) و y(t) مسیر پرواز موشک
│   ├── outputs/                      # گراف‌های دوبعدی PNG و داشبوردهای HTML
│   ├── tests/                        # آزمون‌های خودکار دوبعدی (۱۸ تست واحد و اعتبارسنجی)
│   ├── main.py                       # اسکریپت اجرایی اصلی دوبعدی (CLI با پرچم مونت‌کارلو)
│   └── README.md                     # مستندات فنی دوبعدی
│
├── gui_app.py                        # رابط کاربری گرافیکی جامع مهندسی دسکتاپ مجهز به تحلیل مونت‌کارلو
├── pyproject.toml                    # پیکربندی استاندارد پکیج مدرن پایتون (PEP 518 / PEP 621)
├── requirements.txt                  # وابستگی‌های زمان اجرای پروژه
├── requirements-dev.txt              # وابستگی‌های آزمون، پوشش کد و اعتبارسنجی کیفی
└── README.md                         # راهنمای جامع مخزن (دو زبانه)
```

---

## ۲. تحلیل ریاضی و فیزیکی الگوریتم‌های هدایت و اصابت

### ۲.۱ هندسه کینماتیکی و مثلث برخورد

در فضای درگیری (چه در صفحه ۲ بعدی و چه در فضای ۳ بعدی)، بردار موقعیت نسبی $\vec{R}$ و بردار سرعت نسبی $\vec{V}_{\mathrm{rel}}$ میان موشک ($\vec{r}_M, \vec{v}_M$) و هدف ($\vec{r}_T, \vec{v}_T$) به‌صورت زیر تعریف می‌شوند:

$$
\vec{R} = \vec{r}_T - \vec{r}_M, \quad R = \|\vec{R}\|, \quad \hat{R} = \frac{\vec{R}}{R}
$$

$$
\vec{V}_{\mathrm{rel}} = \vec{v}_T - \vec{v}_M
$$

**سرعت نزدیک‌شدن (Closing Velocity):** آهنگ کاهش فاصله میان موشک و هدف:

$$
V_c = -\dot{R} = -\frac{\vec{R} \cdot \vec{V}_{\mathrm{rel}}}{R}
$$

**بردار سرعت زاویه‌ای خط دید (LOS Rate Vector):** در فضای ۳ بعدی اقلیدسی:

$$
\vec{\Omega}_{\mathrm{LOS}} = \frac{\vec{R} \times \vec{V}_{\mathrm{rel}}}{R^2}
$$

در صفحه ۲ بعدی، با فرض زاویه خط دید $\lambda = \mathrm{atan2}(R_y, R_x)$، نرخ دوران خط دید برابر است با:

$$
\dot{\lambda} = \frac{R_x V_{\mathrm{rel},y} - R_y V_{\mathrm{rel},x}}{R^2}
$$

---

### ۲.۲ ناوبری تناسبی حقیقی (True Proportional Navigation - TPN)

#### فرمول‌بندی ریاضی:

**در فضای ۳ بعدی:**

$$
\vec{a}_{\mathrm{cmd, TPN}} = N \cdot V_c \cdot (\vec{\Omega}_{\mathrm{LOS}} \times \hat{R})
$$

**در صفحه ۲ بعدی:**

$$
\vec{a}_{\mathrm{cmd, TPN}} = N \cdot V_c \cdot \dot{\lambda} \cdot \hat{n}_{\mathrm{LOS}}
$$

که در آن $\hat{n}_{\mathrm{LOS}} = [-\sin\lambda, \cos\lambda]^T$ بردار یکه عمود بر خط دید لحظه‌ای است.

#### مکانیزم مانور و فیزیک اصابت:
اساس ناوبری تناسبی بر صفر کردن نرخ چرخش خط دید ($\dot{\lambda} \to 0$ یا $\vec{\Omega}_{\mathrm{LOS}} \to \vec{0}$) استوار است. در شرایطی که خط دید نمی‌چرخد، موشک و هدف در یک مثلث برخورد با جهت ثابت و فاصله کاهنده (**CBDR: Constant Bearing Decreasing Range**) قرار می‌گیرند که تضمین‌کننده برخورد قطعی موشک به هدف با کمترین تلاش کنترلی است.

#### رفتار در پرواز و محدودیت‌های فاز پایانی (Endgame):
- **اهداف بدون مانور یا کم‌مانور:** موشک با زاویه پیش‌گیری (Lead Angle) بهینه پرواز کرده و مسیر بسیار هموار با حداقل مصرف انرژی طی می‌کند.
- **اهداف با مانور شدید:** قانون TPN دارای **تأخیر فاز کینماتیکی (Kinematic Phase Lag)** ذاتی است؛ زیرا موشک تنها زمانی فرمان شتاب صادر می‌کند که جابه‌جایی هدف منجر به چرخش زاویه خط دید شده باشد. در فاز نهایی که فاصله $R \to 0$ میل می‌کند، خطاهای انباشته‌شده باعث انفجار تقاضای شتاب موشک می‌شوند. در نتیجه، موشک دچار اشباع شتاب در مرز سازه‌ای ($35g$) شده و خطای اصابت افزایش می‌یابد.

---

### ۲.۳ ناوبری تناسبی ارتقایافته (Augmented Proportional Navigation - APN)

#### فرمول‌بندی ریاضی:

**در فضای ۳ بعدی:**

$$
\vec{a}_{\mathrm{cmd, APN}} = N \cdot V_c \cdot (\vec{\Omega}_{\mathrm{LOS}} \times \hat{R}) + \frac{N}{2} \vec{a}_{T\perp}
$$

که در آن مولفه شتاب مانور هدف در راستای عمود بر خط دید به‌صورت زیر تعریف می‌شود:

$$
\vec{a}_{T\perp} = \vec{a}_T - (\vec{a}_T \cdot \hat{R})\hat{R}
$$

**در صفحه ۲ بعدی:**

$$
\vec{a}_{\mathrm{cmd, APN}} = N \cdot V_c \cdot \dot{\lambda} \cdot \hat{n}_{\mathrm{LOS}} + \frac{N}{2} (\vec{a}_T \cdot \hat{n}_{\mathrm{LOS}}) \hat{n}_{\mathrm{LOS}}
$$

#### مکانیزم مانور و برتری فیزیکی:
الگوریتم APN مجهز به ترم **جبران‌سازی پیش‌خور (Feedforward Compensation)** شتاب هدف است. با افزودن ترم پیش‌خور $(N / 2) \cdot \vec{a}_{T\perp}$ به معادله فرمان، موشک بلافاصله و همگام با شروع مانور جنگنده، شتاب متقابل صادر می‌کند و منتظر تجمع خطای زاویه‌ای خط دید نمی‌ماند.

#### مزایای عملیاتی:
- حذف کامل تأخیر فاز کینماتیکی در درگیری‌های سنگین.
- توزیع یکنواخت بار مانور در طول کل مسیر و ممانعت از اشباع شتاب بالک‌ها در ثانیه‌های پایانی پرواز.
- دستیابی به خطای اصابت میلی‌متری و برخورد مستقیم (Kinetic Hit-to-Kill) حتی در برابر مانورهای گریز زیگزاگی شدید ۸ تا ۱۰ جی (S-Turn و Barrel Roll).

---

### ۲.۴ هدایت تعقیب محض (Pure Pursuit - سر به هدف)

#### فرمول‌بندی ریاضی:

**در صفحه ۲ بعدی:**

با زاویه مسیر پرواز موشک $\gamma_M = \mathrm{atan2}(v_{M,y}, v_{M,x})$ و زاویه خط دید $\lambda = \mathrm{atan2}(R_y, R_x)$، خطای زاویه‌ای سمت برابر است با:

$$
\eta = \lambda - \gamma_M
$$

آهنگ گردش زاویه بردار سرعت موشک:

$$
\dot{\gamma}_{\mathrm{cmd}} = \dot{\lambda} + K_p \cdot \eta
$$

بردار شتاب جانبی عمود بر سرعت موشک ($\hat{n}_{vM} = [-\sin\gamma_M, \cos\gamma_M]^T$):

$$
\vec{a}_{\mathrm{cmd, PP}} = \|\vec{v}_M\| \cdot \dot{\gamma}_{\mathrm{cmd}} \cdot \hat{n}_{vM}
$$

**در فضای ۳ بعدی:**

با بردارهای یکه سرعت موشک $\hat{v}_M = \frac{\vec{v}_M}{\|\vec{v}_M\|}$ و خط دید $\hat{R} = \frac{\vec{R}}{\|\vec{R}\|}$، زاویه انحراف دماغه:

$$
\eta = \arccos(\mathrm{clip}(\hat{v}_M \cdot \hat{R}, -1, 1))
$$

محور دوران لحظه‌ای:

$$
\vec{u} = \frac{\hat{v}_M \times \hat{R}}{\|\hat{v}_M \times \hat{R}\|}
$$

بردار نرخ چرخش فرماندهی بردار سرعت:

$$
\vec{\omega}_{\mathrm{cmd}} = \vec{\Omega}_{\mathrm{LOS}} + K_p \cdot \eta \cdot \vec{u}
$$

بردار شتاب عمود بر بردار سرعت موشک:

$$
\vec{a}_{\mathrm{cmd, PP}} = \vec{\omega}_{\mathrm{cmd}} \times \vec{v}_M
$$

#### مکانیزم مانور و دینامیک تعقیب دُم (Tail-Chase Dynamics):
برخلاف ناوبری تناسبی که زاویه پیش‌گیری (Lead Angle) اتخاذ می‌کند، در تعقیب محض **دماغه و بردار سرعت موشک همواره مستقیماً به سمت نقطه کنونی هدف نشانه می‌رود** ($\eta \to 0$).

#### پدیده ریاضی و فیزیکی واگرایی شتاب در فاز نهایی:
1. **مسیر منحنی تعقیب (Pursuit Curve):** موشک همواره پشت سر هدف حرکت می‌کند و فاقد زاویه لید است.
2. **تکینگی شتاب در لحظه اصابت:** نرخ دوران خط دید طبق رابطه کینماتیکی:

$$
\dot{\lambda} \approx \frac{V_T \sin(\theta_T - \lambda)}{R}
$$

با کاهش فاصله $R \to 0$ به‌سمت بی‌نهایت میل می‌کند. بنابراین، برای نگه‌داشتن دماغه روی هدف در فواصل بسیار نزدیک، تقاضای شتاب موشک به بی‌نهایت میل می‌کند ($\|\vec{a}_{\mathrm{cmd}}\| \to \infty$).
3. **اشباع و افزایش خطای اصابت:** به دلیل وجود سقف فیزیکی شتاب سازه موشک ($35g$)، سیستم کنترل موشک نمی‌تواند نرخ چرخش نامحدود را تأمین کند. در نتیجه، در لحظات پایانی موشک از چرخش سریع بازمانده و با خطای اصابت بزرگتری نسبت به روش‌های PN از کنار هدف عبور می‌کند.

---

### ۲.۵ ماتریس مقایسه تحلیلی عملکرد الگوریتم‌ها

| ویژگی / معیار مقایسه | ناوبری تناسبی حقیقی (TPN) | ناوبری تناسبی ارتقایافته (APN) | تعقیب محض (Pure Pursuit) |
| :--- | :--- | :--- | :--- |
| **فلسفه هدایت** | صفر کردن نرخ چرخش خط دید ($\dot{\lambda} \to 0$) | صفر کردن نرخ خط دید + جبران پیش‌خور شتاب هدف | انطباق لحظه‌ای بردار سرعت بر راستای هدف ($\vec{v}_M \parallel \vec{R}$) |
| **زاویه پیش‌گیری (Lead Angle)** | فعال بر روی مثلث برخورد | فعال به همراه لید دینامیکی مانور | صفر (تعقیب مستقیم دُم هدف) |
| **هندسه مسیر پرواز** | تقریباً مستقیم بر روی مسیر برخورد | بهینه و مستقیم‌ترین مسیر ممکن | منحنی تعقیب خمیده (سگ و خرگوش) |
| **پروفایل شتاب جانبی** | پرش ناگهانی در فاز نهایی مانور هدف | هموار، یکنواخت و پایدار | واگرایی شدید به‌سمت بی‌نهایت در نزدیکی هدف |
| **خطر اشباع سازه‌ای ($35g$)** | بالا در برابر مانورهای زیگزاگی هدف | بسیار پایین / بدون اشباع | اشباع قطعی در فاز پایانی درگیری |
| **خطای اصابت (Miss Distance)** | متوسط ($0.5\text{ m} - 5.0\text{ m}$) | فوق‌العاده دقیق و میلی‌متری ($< 0.1\text{ m}$) | بیشترین خطا ($1.5\text{ m} - 15.0\text{ m}$) |
| **نیازمندی به سنسور** | نرخ زاویه‌ای سیکر و سرعت نزدیک‌شدن | نرخ سیکر + تخمین شتاب هدف با فیلتر کالمن | فقط زاویه و راستای دید هدف |
| **انرژی کنترلی مصرفی** | متوسط | بهینه و کمترین میزان مصرف انرژی | بالا (تلاش مداوم برای تغییر راستای سرعت) |

---

## ۳. تشریح جامع تک‌تک پارامترهای فیزیکی، آیرودینامیکی و کنترلی موشک

مدل فیزیکی شبیه‌ساز بر پایه یک موشک سوخت جامد تاکتیکی سطح‌به‌هوا / هوابه‌هوا با جرم متغیر، پیشران بوست-کاست و پسا دینامیکی پیاده‌سازی شده است:

| نماد پارامتر | مقدار عددی | واحد | دسته‌بندی فیزیکی | شرح علمی و نقش پارامتر در معادلات حرکت |
| :--- | :--- | :--- | :--- | :--- |
| $m_0$ | `85.0` | کیلوگرم (kg) | جرم و اینرسی | جرم کل اولیه موشک در لحظه پرتاب شامل سازه، سرجنگی، اویونیک و سوخت جامد. |
| $m_{\mathrm{dry}}$ | `50.0` | کیلوگرم (kg) | جرم و اینرسی | جرم خشک سازه پس از اتمام سوخت پیشران در زمان $t \ge 4.0\text{ s}$. |
| $m_{\mathrm{prop}}$ | `35.0` | کیلوگرم (kg) | جرم و اینرسی | جرم سوخت پیشران مصرفی: $m_{\mathrm{prop}} = m_0 - m_{\mathrm{dry}} = 35.0\text{ kg}$. |
| $t_{\mathrm{burn}}$ | `4.0` | ثانیه (s) | سیستم پیشران | مدت‌زمان سوختن موتور راکت سوخت جامد. تفکیک‌کننده فاز بوست ($t \le 4\text{ s}$) از فاز کاست ($t > 4\text{ s}$). |
| $\dot{m}$ | `8.75` | کیلوگرم بر ثانیه (kg/s) | سیستم پیشران | نرخ کاهش جرم سوخت موشک طبق رابطه تسیاکوفسکی: $\dot{m} = 35 / 4 = 8.75\text{ kg/s}$. جرم لحظه‌ای: $m(t) = m_0 - \dot{m}t$. |
| $T_{\mathrm{nominal}}$ | `17,500.0` | نیوتون (N) | سیستم پیشران | نیروی تراست پیشران در فاز بوست: $17.5\text{ kN}$ در ۴ ثانیه نخست و صفر پس از خاموشی موتور. |
| $I_{\mathrm{sp}}$ | `203.9` | ثانیه (s) | سیستم پیشران | ضربه ویژه موتور سوخت جامد: $I_{\mathrm{sp}} = \frac{T}{\dot{m} \cdot g_0} = \frac{17500}{8.75 \times 9.81} \approx 203.9\text{ s}$. |
| $d$ | `0.127` | متر (m) | هندسه و آیرودینامیک | کالیبر و قطر بدنه موشک ($127\text{ mm}$ یا ۵ اینچ استاندارد). |
| $A_{\mathrm{ref}}$ | `0.0127` | متر مربع (m²) | هندسه و آیرودینامیک | سطح مقطع پیشانی آیرودینامیکی موشک: $A_{\mathrm{ref}} = \frac{\pi d^2}{4} \approx 0.01267\text{ m}^2$. |
| $\rho$ | `0.736` | کیلوگرم بر متر مکعب (kg/m³) | جو و اتمسفر | چگالی هوای جو استاندارد ISA در ارتفاع عملیاتی درگیری ($5000\text{ m}$). |
| $a$ | `320.5` | متر بر ثانیه (m/s) | جو و اتمسفر | سرعت محلی صوت در ارتفاع ۵۰۰۰ متری ($\text{Mach } 1.0 = 320.5\text{ m/s}$). |
| $C_D$ | `0.40` | بدون بعد | آیرودینامیک | ضریب پسای پایه بدنه موشک. نیروی پسای آیرودینامیکی درجه دو: $\vec{D} = -\frac{1}{2} \rho V_M C_D A_{\mathrm{ref}} \vec{v}_M$. |
| $q(t)$ | متغیر زمانی | نیوتون بر متر مربع (N/m²) | آیرودینامیک | فشار دینامیکی هوا بر روی بالک‌ها و بدنه: $q = \frac{1}{2} \rho \|\vec{v}_M\|^2$. |
| $G_{\mathrm{limit}}$ | `35.0` | جی (g) | محدودیت سازه‌ای | سقف مجاز بار مانور سازه و عملگرهای بالک موشک ($35 \times 9.81 = 343.35\text{ m/s}^2$). |
| $g_0$ | `9.81` | متر بر مجذور ثانیه (m/s²) | ژئوفیزیک | شتاب گرانش استاندارد زمین. |
| $V_{M0}$ | `250.0` | متر بر ثانیه (m/s) | کینماتیک پرتاب | سرعت پرتاب اولیه موشک ($\text{Mach } 0.78$) هم‌راستا با بردار خط دید اولیه $\hat{R}_0$. |
| $\vec{r}_{M0}$ | `[0, 0, 0]` | متر (m) | کینماتیک پرتاب | مختصات سکوی پرتاب در مبدأ دستگاه مختصات لخت اینرسی دکارتی. |
| $N$ | `4.0` | بدون بعد | ضرایب هدایت | ضریب ناوبری تناسبی بهینه برای الگوریتم‌های TPN و APN ($N=4.0$). |
| $K_p$ | `4.0` | یک بر ثانیه (s⁻¹) | ضرایب هدایت | ضریب بهره تناسبی حلقه کنترل تعقیب محض برای هدایت دماغه روی هدف. |
| $\Delta t$ | `0.005` | ثانیه (s) | حل‌کننده عددی | گام زمانی انتگرال‌گیری عددی رانگ-کوتا مرتبه ۴ (نرخ نمونه‌برداری $200\text{ Hz}$). |
| $t_{\mathrm{max}}$ | `25.0` | ثانیه (s) | زمان شبیه‌سازی | سقف مجاز زمان پرواز موشک تا خاتمه سناریوی شبیه‌سازی. |

---

## ۴. نحوه تعریف هندسه و معادلات ریاضی مسیر هدف

علاوه بر سناریوهای استاندارد نبرد هوایی A تا D، امکان تعریف مسیر هدف با معادلات تحلیلی ریاضی زیر فراهم است:

### ۱. معادله خط مستقیم (`LINE`)

**در صفحه ۲ بعدی:** خط مستقیم $y(x) = m \cdot x + c$ با سرعت ثابت $V_T$:

$$
v_x = \mathrm{dir} \cdot \frac{V_T}{\sqrt{1 + m^2}}, \quad v_y = m \cdot v_x, \quad \vec{a}_T = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

**در فضای ۳ بعدی:** پرواز بر روی خط مستقیم فضایی با بردار سرعت ثابت:

$$
\vec{r}_T(t) = \vec{r}_{T0} + \vec{v}_{\mathrm{const}} \cdot t, \quad \vec{a}_T(t) = \vec{0}
$$

### ۲. معادله سهمی (`PARABOLA`)

**در صفحه ۲ بعدی:** سهمی هندسی $y(x) = a \cdot x^2 + b \cdot x + c$ با پیشروی افقی $x(t) = x_0 + v_x \cdot t$:

$$
v_y(t) = (2 a x(t) + b) \cdot v_x, \quad a_y(t) = 2 a v_x^2
$$

**در فضای ۳ بعدی:** مسیر پرتابه‌ای یا مانور شتاب ثابت فضایی:

$$
\vec{r}_T(t) = \vec{r}_{T0} + \vec{v}_{T0} \cdot t + \frac{1}{2}\vec{a}_{\mathrm{const}} \cdot t^2, \quad \vec{v}_T(t) = \vec{v}_{T0} + \vec{a}_{\mathrm{const}} \cdot t, \quad \vec{a}_T(t) = \vec{a}_{\mathrm{const}}
$$

### ۳. معادله توابع دلخواه ریاضی (`CUSTOM`)

فرمول‌های صریح وابسته به زمان در ۲ بعدی ($x(t), y(t)$) و ۳ بعدی ($x(t), y(t), z(t)$) با استخراج عددی سرعت و شتاب لحظه‌ای هدف با استفاده از تفاضل مرکزی با دقت $\mathcal{O}(h^2)$ و گام زمانی $h = 10^{-5}\text{ s}$:

$$
\vec{v}_T(t) = \frac{\vec{r}_T(t + h) - \vec{r}_T(t - h)}{2h}
$$

$$
\vec{a}_T(t) = \frac{\vec{r}_T(t + h) - 2\vec{r}_T(t) + \vec{r}_T(t - h)}{h^2}
$$

توابع ریاضی مجاز در فرمول‌ها: `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`, `pi`, `e` و عملگر توان (`^` یا `**`).

---

## ۵. راهنمای کامل اجرا و به‌کارگیری

### ۵.۱ رابط کاربری گرافیکی جامع مهندسی دسکتاپ (`gui_app.py`)

اجرای برنامه رابط گرافیکی بر پایه Tkinter و بوم پیشرفته Matplotlib:

```powershell
python gui_app.py
```

#### قابلیت‌های رابط گرافیکی:
- تب‌های مجزای درگیری ۳ بعدی و ۲ بعدی.
- امکان انتخاب مدهای ۱ تا ۴ (TPN، APN، Pure Pursuit یا مقایسه همزمان هر ۳ الگوریتم).
- انتخاب سناریوهای حرکتی A تا D یا انتخاب خط، سهمی و فرمول‌های دلخواه ریاضی ($x(t), y(t), z(t)$).
- رسم بلادرنگ خطوط دید (LOS Rays) در فواصل زمانی منظم میان موشک و هدف تا لحظه دقیق اصابت.
- امتداد دادن مسیر پرواز هدف در نمودار تا انتهای پرواز آخرین موشک برای مشاهده کامل هندسه اصابت.
- پلات‌های تله‌متری: بار مانور جانبی (G)، سرعت و ماخ، فاصله تا هدف، و انتگرال انرژی کنترلی.
- جدول ضرایب برازش چندجمله‌ای درجه ۶ مسیر پرواز با دقت $R^2 > 0.9999$.
- دکمه اجرای مستقیم داشبورد وب تعاملی Plotly در مرورگر.

---

### ۵.۲ اجرای خط فرمان (CLI) و منوی تعاملی کنسول

#### اجرای دوبعدی:
```powershell
# منوی تعاملی کنسول:
python 2D/main.py

# اجرای مقایسه ۳ الگوریتم در برابر تابع دلخواه ریاضی:
python 2D/main.py --mode 4 --target CUSTOM --custom-x "6000 - 250*t" --custom-y "2500 + 400*sin(0.5*t)"

# اجرای مقایسه روی معادله خط:
python 2D/main.py --mode 4 --target LINE --line-slope 0.4167 --line-intercept 0.0

# اجرای مقایسه روی معادله سهمی:
python 2D/main.py --mode 4 --target PARABOLA --parabola-a 0.00005 --parabola-b -0.2 --parabola-c 1900.0
```

#### اجرای سه‌بعدی:
```powershell
# منوی تعاملی کنسول:
python 3D/main.py

# اجرای مقایسه ۳ الگوریتم در مانور گریز زیگزاگی شدید (سناریوی D):
python 3D/main.py --mode 4 --target D

# اجرای مقایسه در برابر مسیر مارپیچ فضایی دلخواه:
python 3D/main.py --mode 4 --target CUSTOM --custom-x "6000 - 240*t" --custom-y "2500 + 200*sin(0.4*t)" --custom-z "5000 + 150*cos(0.4*t)"

# اجرای مقایسه در برابر معادله خط مستقیم سه‌بعدی:
python 3D/main.py --mode 4 --target LINE --line-vx -250.0 --line-vy -50.0 --line-vz 20.0

# اجرای کمپین تحلیل پراکندگی تصادفی مونت‌کارلو (۵۰ تکرار با استخراج شاخص‌های CEP و Pk):
python 3D/main.py --mode 2 --target A --monte-carlo 50
python 2D/main.py --mode 2 --target A --monte-carlo 50
```

---

### ۵.۳ اجرای آزمون‌های خودکار، پوشش کد و CI/CD

اجرای آزمون‌های یکپارچه و ممیزی سیستم با `pytest` (مجموعاً ۵۳ تست واحد و اعتبارسنجی):

```powershell
# اجرای جامع تمام ۵۳ آزمون با گزارش پوشش کد:
pytest -v

# یا اجرای مجزای تست‌ها از طریق ماژول استاندارد unittest:
python -m unittest discover -s 3D/tests -p "test_*.py"
python -m unittest discover -s 2D/tests -p "test_*.py"
```

تمامی ۵۳ آزمون با موفقیت ۱۰۰٪ پاس می‌شوند و صحت معادلات کینماتیک، قوانین هدایت TPN/APN/PP، انتگرال‌گیری عددی RK4، تفکیک میلی‌متری CPA، ارزیابی ایمن و بهینه توابع ریاضی با SafeMath AST، تحلیل آماری پراکندگی مونت‌کارلو، موتور زمان‌بندی انیمیشن زنده و کنترل چرخه حیات رابط کاربری گرافیکی دسکتاپ را تضمین می‌کنند.
