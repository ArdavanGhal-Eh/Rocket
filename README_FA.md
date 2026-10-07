<a id="readme-top"></a>

<div align="center">

[![English Documentation](https://img.shields.io/badge/Documentation-English-blue.svg?style=for-the-badge)](README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Rust Accelerated](https://img.shields.io/badge/Rust-Core_Engine-DEA584.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![Simulations](https://img.shields.io/badge/Simulation-6DOF_Rigid_Body-red.svg?style=for-the-badge)](https://github.com/ArdavanGhal-Eh/Rocket)

<br />

# 🚀 شبیه‌ساز دینامیک پرواز و بالستیک شش درجه آزادی راکت (6-DOF Rocket Trajectory Flight Dynamics Simulator)
### *حل عددی معادلات دیفرانسیل غیرخطی نیوتن-اویلر، آیرودینامیک باریکه‌بندی‌شده غیرپایا، مدلسازی پیشران سوخت جامد و تحلیل تفرق مونت‌کارلو*

<p align="center">
  <b>یک چارچوب محاسباتی و چندفیزیکی جامع برای شبیه‌سازی پرواز راکت‌های زیرمداری و کاوشگر در شش درجه آزادی فضایی ($6\text{-DOF}$). این بستر ترکیبی از موتور یکپارچه‌ساز عددی گام‌تطبیقی رونگه-کوتا مرتبه چهار ($RK4$)، مدل جو استاندارد بین‌المللی ($ISA\text{-}1976$)، استخراج ضرایب آیرودینامیکی بارویکمن (Barrowman Method)، تحلیل نوسان جرم متغیر در اثر سوزش گرین پیشران و توزیع تصادفی باد مونت‌کارلو است.</b>
  <br /><br />
  <a href="#-مبانی-نظری-و-فرمولاسیون-ریاضی"><strong>معادلات دینامیک پرواز »</strong></a>
  &nbsp;•&nbsp;
  <a href="#-معماری-نرم‌افزاری-شبیه‌ساز"><strong>معماری شبیه‌ساز »</strong></a>
  &nbsp;•&nbsp;
  <a href="#-نحوه-اجرا-و-آزمون‌ها"><strong>راهنمای اجرا و خروجی‌ها »</strong></a>
</p>

</div>

---

<details open>
  <summary><h2 style="display: inline-block;">📑 فهرست مطالب</h2></summary>
  <ol>
    <li><a href="#-چکیده-مهندسی-هوافضا">چکیده مهندسی هوافضا</a></li>
    <li><a href="#-قابلیت‌های-محاسباتی-شبیه‌ساز">قابلیت‌های محاسباتی شبیه‌ساز</a></li>
    <li><a href="#-مبانی-نظری-و-فرمولاسیون-ریاضی">مبانی نظری و فرمولاسیون ریاضی</a></li>
    <li><a href="#-آیرودینامیک-و-مدل-اتمسفر">آیرودینامیک و مدل اتمسفر</a></li>
    <li><a href="#-مدل-پیشران-و-تغییر-جرم-ناپایا">مدل پیشران و تغییر جرم ناپایا</a></li>
    <li><a href="#-تحلیل-پراکندگی-مونت‌کارلو-monte-carlo">تحلیل پراکندگی مونت‌کارلو (Monte Carlo)</a></li>
    <li><a href="#-معماری-نرم‌افزاری-شبیه‌ساز">معماری نرم‌افزاری شبیه‌ساز</a></li>
    <li><a href="#-نحوه-اجرا-و-آزمون‌ها">نحوه اجرا و آزمون‌ها</a></li>
    <li><a href="#-توسعه-آتی-و-همکاری">توسعه آتی و همکاری</a></li>
    <li><a href="#-مجوز">مجوز</a></li>
  </ol>
</details>

---

## 📌 چکیده مهندسی هوافضا

شبیه‌سازی دینامیک پرواز راکت نیازمند کوپل همزمان پدیده‌های چندرشته‌ای است: کاهش پیوسته جرم ناشی از خروج گازهای حاصل از احتراق، تغییر لحظه‌ای مرکز جرم ($CG$) و تانسور ممان اینرسی ($I_{xx}, I_{yy}, I_{zz}$)، جابجایی مرکز فشار آیرودینامیکی ($CP$) در رژیم‌های مختلف عدد ماخ ($Mach$) و وزش بادهای لایه‌ای متلاطم اتمسفر. 

این شبیه‌ساز معادلات حرکت جسم صلب را با استفاده از متغیرهای کواترنیون یکانی ($Unit Quaternions$) فرموله نموده تا از پدیده قفل گیمبال ($Gimbal Lock$) در زوایای تقرب عمودی جلوگیری کند.

---

## 🚀 قابلیت‌های محاسباتی شبیه‌ساز

- **شبیه‌سازی کامل شش درجه آزادی ($6\text{-DOF}$):** حل همزمان ۳ معادله انتقال انتقالی ($Translational$) و ۳ معادله دوران دورانی ($Rotational$).
- **کواترنیون‌های اویلر برای جلوگیری از قفل گیمبال:** یکپارچه‌سازی سینماتیک وضعیت زاویه‌ای با استفاده از بردار کواترنیون $\mathbf{q} = [q_0, q_1, q_2, q_3]^T$.
- **روش اجزای بارویکمن (Barrowman Component Method):** تخمین مشتقات پایداری استاتیکی ($C_{N\alpha}$, $C_{m\alpha}$) و تعیین فاصله حاشیه استاتیکی ($Static Margin$).
- **حلگر عددی RK4 گام‌متغیر:** پیشروی زمانی دقیق متغیرهای حالت بدون اتلاف انرژی محاسباتی.
- **تحلیل تفرق مونت‌کارلو ($10,000+$ پرواز):** بررسی تاثیر خطاهای زاویه سکوی پرتاب، عدم قطعیت تراست موتور و وزش باد تصادفی گوسی بر نقطه فرود چتر.

---

## 📐 مبانی نظری و فرمولاسیون ریاضی

### ۱. معادلات دینامیک انتقالی (دستگاه مختصات بدنه)
معادلات حرکت انتقالی در دستگاه متصل به بدنه ($Body Frame$):
$$m \left( \frac{d\mathbf{v}_b}{dt} + \boldsymbol{\omega}_b \times \mathbf{v}_b \right) = \mathbf{F}_{\text{aero}} + \mathbf{F}_{\text{thrust}} + \mathbf{F}_{\text{gravity}}$$

### ۲. معادلات دینامیک دورانی اویلر با جرم متغیر
با در نظر گرفتن تغییرات زمانی تانسور اینرسی $\mathbf{I}(t)$:
$$\mathbf{I}(t) \frac{d\boldsymbol{\omega}_b}{dt} + \boldsymbol{\omega}_b \times (\mathbf{I}(t) \boldsymbol{\omega}_b) + \frac{d\mathbf{I}}{dt} \boldsymbol{\omega}_b = \mathbf{M}_{\text{total}}$$

### ۳. سینماتیک کواترنیون
نرخ تغییرات کواترنیون زاویه‌ای با سرعت زاویه‌ای بدنه $\boldsymbol{\omega}_b = [p, q, r]^T$:
$$\begin{bmatrix} \dot{q}_0 \\ \dot{q}_1 \\ \dot{q}_2 \\ \dot{q}_3 \end{bmatrix} = \frac{1}{2} \begin{bmatrix} 0 & -p & -q & -r \\ p & 0 & r & -q \\ q & -r & 0 & p \\ r & q & -p & 0 \end{bmatrix} \begin{bmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{bmatrix}$$

---

## 🌪 آیرودینامیک و مدل اتمسفر

- **مدل اتمسفر استاندارد ۱۹۷۶ (U.S. Standard Atmosphere 1976):** محاسبه تغییرات چگالی $\rho(h)$، فشار $P(h)$ و دمای لایه‌های تروپوسفر و استراتوسفر تا ارتفاع $84\text{ km}$.
- **نیروهای آیرودینامیکی بدنه:**
  $$\mathbf{F}_{\text{aero}} = \frac{1}{2} \rho V_{\infty}^2 S_{\text{ref}} \begin{bmatrix} -C_A \\ -C_N \sin\phi \\ -C_N \cos\phi \end{bmatrix}$$
  که در آن $C_A$ ضریب نیروی محوری و $C_N$ ضریب نیروی عمودی است.

---

## 🎯 تحلیل پراکندگی مونت‌کارلو (Monte Carlo)

برای ارزیابی شعاع ایمنی پرتاب، عدم قطعیت‌های فیزیکی زیر به صورت تصادفی مدل می‌شوند:
- زاویه شیب پرتابه‌گیر: $\theta_0 \sim \mathcal{N}(\mu=85^\circ, \sigma=0.5^\circ)$
- خطای ضربه کل پیشران: $I_{\text{tot}} \sim \mathcal{N}(\mu=I_0, \sigma=0.03 I_0)$
- زاویه و سرعت وزش باد سطحی: مدل پروفیل باد توبولنت ون کارمن (Von Kármán Wind Turbulence).

خروجی شبیه‌سازی یک بیضی پراکندگی برخورد ($Landing Dispersion Ellipse$) با اطمینان ۹۵٪ در مختصات جغرافیایی است.

---

## 🏗 معماری نرم‌افزاری شبیه‌ساز

```
[Configuration & Rocket Parameters] (JSON/YAML)
               │
               ▼
   [Atmosphere Engine] (ISA-1976 / Altitude Layers)
               │
   [Propulsion Engine] (Thrust Curve, Grain Consumption, CG/MOI Shift)
               │
   [Aerodynamics Engine] (Barrowman Linear + Transonic Drag Wave)
               │
               ▼
     [RK4 6-DOF Integrator] ◄─── (State Vectors: Pos, Vel, Quat, Omega)
               │
               ▼
[Trajectory Logging & Event Manager] (Apogee Detection, Parachute Deploy)
               │
               ▼
   [Visualization & Analysis Suite] (3D Plots, Telemetry Curves, Dispersion Map)
```

---

## ⚙️ نحوه اجرا و آزمون‌ها

### نصب وابستگی‌ها
```bash
git clone https://github.com/ArdavanGhal-Eh/Rocket.git
cd Rocket
pip install -r requirements.txt
```

### اجرای شبیه‌سازی پرواز پایه
```bash
python run_simulation.py --config configs/sounding_rocket.json --plot-trajectory
```

### اجرای تحلیل مونت‌کارلو
```bash
python monte_carlo.py --samples 5000 --output reports/dispersion_ellipse.png
```

---

## 📄 مجوز
این پروژه تحت مجوز [MIT](https://opensource.org/licenses/MIT) منتشر شده است.
