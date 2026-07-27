# Reward Formulation for Multi-Objective HVAC Control

This document describes the scalar reward used for reinforcement learning of building HVAC control. All primary terms are expressed in **dollars** (or optionally **dollars per square metre**) so that energy cost, thermal-comfort productivity loss, and indoor-air-quality (IAQ) productivity loss share a common unit. The formulation matches the implementation in `tests/rl_hvac_control.py` with configuration in `config/hvac_config.yaml`.

---

## 1. Scalar reward

At each control timestep \(t\) with duration \(\Delta t\) hours, the agent receives

$$
r_t = -\, C_t,
$$

where \(C_t\) is the total operating cost for that step. With optional floor-area normalization,

$$
C_t =
\begin{cases}
\dfrac{1}{A}\, C^{\mathrm{abs}}_t, & \text{if cost normalization mode is } \texttt{per\_m2}, \\[0.6em]
C^{\mathrm{abs}}_t, & \text{otherwise (absolute dollars)}.
\end{cases}
$$

Here \(A\) is the conditioned floor area (m\(^2\)); for the DOE Medium Office prototype, \(A = 4982\). The absolute cost is

$$
C^{\mathrm{abs}}_t
=
C^{\mathrm{elec}}_t
+ C^{\mathrm{gas}}_t
+ C^{\mathrm{th}}_t
+ C^{\mathrm{iaq}}_t
+ C^{\mathrm{sp}}_t
+ C^{\mathrm{dem}}_t.
$$

In the default configuration, setpoint and demand terms are disabled (\(C^{\mathrm{sp}}_t = C^{\mathrm{dem}}_t = 0\)), so

$$
C^{\mathrm{abs}}_t
=
C^{\mathrm{elec}}_t
+ C^{\mathrm{gas}}_t
+ C^{\mathrm{th}}_t
+ C^{\mathrm{iaq}}_t.
$$

---

## 2. Electricity cost (real-time price)

Let \(P_t\) be the facility total electric demand (W) and \(p_t\) the real-time electricity price (\$/kWh) at the calendar hour corresponding to \(t\). Then

$$
E_t = \frac{P_t}{1000}\,\Delta t
\quad\text{(kWh)},
\qquad
C^{\mathrm{elec}}_t = E_t\, p_t
\quad\text{(\$)}.
$$

No additional electric weight is applied: the term is already monetized by the tariff.

---

## 3. Gas cost

Let \(G_t\) be gas energy use over the timestep (kWh thermal) and \(p_g\) the gas price (\$/kWh). Then

$$
C^{\mathrm{gas}}_t = G_t\, p_g
\quad\text{(\$)}.
$$

Again, no extra gas weight is used.

---

## 4. Thermal comfort as productivity loss

Thermal discomfort is monetized via zone Predicted Percentage of Dissatisfied (PPD) from the Fanger model in EnergyPlus, translated to a fractional productivity loss and then to dollars through labour value and occupancy.

For each controlled zone \(z\) with occupant count \(N_{z,t}\) and PPD \(PPD_{z,t}\) (percent),

$$
\ell^{\mathrm{th}}_{z,t}
=
\mathrm{clip}\!\left(
\bigl(PPD_{z,t} - \pi_{\mathrm{ref}}\bigr)
\frac{\alpha_{\mathrm{ppd}}}{100},\;
0,\;
\ell^{\mathrm{th}}_{\max}
\right),
$$

where

- \(\pi_{\mathrm{ref}}\) is the PPD reference (default \(10\%\), consistent with typical ASHRAE Standard 55 design intent),
- \(\alpha_{\mathrm{ppd}}\) is the productivity-loss slope per one percentage point of PPD above the reference (default \(0.5\)),
- \(\ell^{\mathrm{th}}_{\max}\) caps fractional loss (default \(0.15\)),
- \(\mathrm{clip}(x;0,u)=\min\{\max\{x,0\},u\}\).

The **raw** thermal productivity cost is

$$
C^{\mathrm{th,raw}}_t
=
\sum_{z}
\ell^{\mathrm{th}}_{z,t}\,
N_{z,t}\,
w_{\mathrm{labor}}\,
\Delta t,
$$

with labour value \(w_{\mathrm{labor}}\) in \$/person/h (default \(40\)). Zones with \(N_{z,t}=0\) contribute nothing.

The term entering the reward is

$$
C^{\mathrm{th}}_t
=
C^{\mathrm{th,raw}}_t
\cdot
w_{\mathrm{c}}
\cdot
s^{\mathrm{th}}_t,
$$

where \(w_{\mathrm{c}}\) is a static comfort weight (default \(1\)) and \(s^{\mathrm{th}}_t\) is an optional online adaptive scale (Section 6).

This mapping is an engineering cost–benefit construction informed by PPD–productivity literature (e.g. Kosonen & Tan; Lan, Wargocki & Lian), not a unique closed-form law from a single study.

---

## 5. IAQ as productivity loss (floor CO\(_2\))

Indoor air quality is represented by floor-average CO\(_2\) concentration. For each floor \(f \in \{\mathrm{bottom},\mathrm{mid},\mathrm{top}\}\) with total occupants \(N_{f,t}\) and mean zone CO\(_2\) \(c_{f,t}\) (ppm),

$$
\ell^{\mathrm{iaq}}_{f,t}
=
\mathrm{clip}\!\left(
\frac{c_{f,t}-c_{\mathrm{ref}}}{100}\,
\alpha_{\mathrm{co2}},\;
0,\;
\ell^{\mathrm{iaq}}_{\max}
\right),
$$

where \(c_{\mathrm{ref}}\) is a reference concentration (default \(800\,\mathrm{ppm}\)), \(\alpha_{\mathrm{co2}}\) is the fractional loss per \(+100\,\mathrm{ppm}\) (default \(0.01\)), and \(\ell^{\mathrm{iaq}}_{\max}\) is a cap (default \(0.15\)).

The raw IAQ cost is

$$
C^{\mathrm{iaq,raw}}_t
=
\sum_{f}
\ell^{\mathrm{iaq}}_{f,t}\,
N_{f,t}\,
w_{\mathrm{labor}}\,
\Delta t,
$$

and the reward term is

$$
C^{\mathrm{iaq}}_t
=
C^{\mathrm{iaq,raw}}_t
\cdot
w_{\mathrm{iaq}}
\cdot
s^{\mathrm{iaq}}_t,
$$

with static weight \(w_{\mathrm{iaq}}\) (default \(1\)) and adaptive scale \(s^{\mathrm{iaq}}_t\) (Section 6).

The CO\(_2\)–performance slope is an order-of-magnitude proxy consistent with ventilation/IAQ–productivity reviews (e.g. Seppänen, Fisk & Lei; Wargocki et al.), using floor CO\(_2\) as a measurable proxy for ventilation effectiveness.

---

## 6. Adaptive cost balancing

Because full-wage productivity losses can dominate electricity cost by one to two orders of magnitude, optional **online balancing** scales the thermal and IAQ terms toward the observed energy cost using exponential moving averages (EMAs) collected during training.

Define the energy reference

$$
C^{\mathrm{E}}_t = C^{\mathrm{elec}}_t + C^{\mathrm{gas}}_t.
$$

With EMA smoothing factor \(\alpha \in (0,1]\),

$$
\begin{aligned}
m^{\mathrm{E}}_t
&=
(1-\alpha)\, m^{\mathrm{E}}_{t-1}
+
\alpha\, C^{\mathrm{E}}_t, \\
m^{\mathrm{th}}_t
&=
(1-\alpha)\, m^{\mathrm{th}}_{t-1}
+
\alpha\, C^{\mathrm{th,raw}}_t, \\
m^{\mathrm{iaq}}_t
&=
(1-\alpha)\, m^{\mathrm{iaq}}_{t-1}
+
\alpha\, C^{\mathrm{iaq,raw}}_t.
\end{aligned}
$$

After a warmup of \(n_{\min}\) samples, the adaptive scales are

$$
\begin{aligned}
s^{\mathrm{th}}_t
&=
\mathrm{clip}\!\left(
\rho_{\mathrm{th}}\,
\frac{m^{\mathrm{E}}_t}{m^{\mathrm{th}}_t},\;
s_{\min},\;
s_{\max}
\right), \\[0.4em]
s^{\mathrm{iaq}}_t
&=
\mathrm{clip}\!\left(
\rho_{\mathrm{iaq}}\,
\frac{m^{\mathrm{E}}_t}{m^{\mathrm{iaq}}_t},\;
s_{\min},\;
s_{\max}
\right),
\end{aligned}
$$

provided the corresponding EMA exceeds a small threshold (otherwise the previous scale is retained). During warmup, \(s^{\mathrm{th}}_t = s^{\mathrm{iaq}}_t = s_0\) (default \(s_0 = 1\)).

Default settings used in this work:

| Symbol | Config key | Default |
|--------|------------|---------|
| \(\alpha\) | `ema_alpha` | \(0.01\) |
| \(n_{\min}\) | `min_samples` | \(100\) |
| \(\rho_{\mathrm{th}}\) | `comfort_target_ratio` | \(1\) |
| \(\rho_{\mathrm{iaq}}\) | `co2_target_ratio` | \(1\) |
| \(s_{\min},\,s_{\max}\) | `weight_min`, `weight_max` | \(0,\,1\) |

With \(s_{\max}=1\), adaptive balancing **only reduces** productivity costs when they dominate energy; it does not amplify them above the static weights \(w_{\mathrm{c}}\) and \(w_{\mathrm{iaq}}\). Target ratios \(\rho=1\) aim for

$$
\mathbb{E}\bigl[C^{\mathrm{th}}_t\bigr]
\approx
\mathbb{E}\bigl[C^{\mathrm{E}}_t\bigr],
\qquad
\mathbb{E}\bigl[C^{\mathrm{iaq}}_t\bigr]
\approx
\mathbb{E}\bigl[C^{\mathrm{E}}_t\bigr]
$$

in the EMA sense (approximately, under slowly varying conditions).

---

## 7. Compact expression (default configuration)

Combining the above with \(\texttt{per\_m2}\) normalization and disabled setpoint/demand penalties,

$$
r_t
=
-\frac{1}{A}
\Bigg[
E_t\, p_t
+
G_t\, p_g
+
w_{\mathrm{c}}\, s^{\mathrm{th}}_t
\sum_{z}
\ell^{\mathrm{th}}_{z,t}\, N_{z,t}\, w_{\mathrm{labor}}\, \Delta t
+
w_{\mathrm{iaq}}\, s^{\mathrm{iaq}}_t
\sum_{f}
\ell^{\mathrm{iaq}}_{f,t}\, N_{f,t}\, w_{\mathrm{labor}}\, \Delta t
\Bigg].
$$

---

## 8. Default numerical parameters

| Quantity | Symbol | Default value |
|----------|--------|---------------|
| Timestep | \(\Delta t\) | \(0.25\,\mathrm{h}\) (15 min) |
| Floor area | \(A\) | \(4982\,\mathrm{m}^2\) |
| Gas price | \(p_g\) | \(0.017\,\mathrm{\$/kWh}\) |
| Labour value | \(w_{\mathrm{labor}}\) | \(40\,\mathrm{\$/person/h}\) |
| PPD reference | \(\pi_{\mathrm{ref}}\) | \(10\%\) |
| PPD loss slope | \(\alpha_{\mathrm{ppd}}\) | \(0.5\) |
| Max thermal loss | \(\ell^{\mathrm{th}}_{\max}\) | \(0.15\) |
| CO\(_2\) reference | \(c_{\mathrm{ref}}\) | \(800\,\mathrm{ppm}\) |
| CO\(_2\) loss slope | \(\alpha_{\mathrm{co2}}\) | \(0.01\) per \(+100\,\mathrm{ppm}\) |
| Max IAQ loss | \(\ell^{\mathrm{iaq}}_{\max}\) | \(0.15\) |
| Static thermal weight | \(w_{\mathrm{c}}\) | \(1\) |
| Static IAQ weight | \(w_{\mathrm{iaq}}\) | \(1\) |

Electricity price \(p_t\) is taken from a real-time price series (CSV) aligned to the simulation calendar; gas and labour prices are treated as constant.

---

## 9. Implementation note

The reward is computed in `HVACEnvironment.compute_reward_components` (`tests/rl_hvac_control.py`). Adaptive scales are maintained by `AdaptiveCostBalancer`. Occupant counts \(N_{z,t}\) and \(N_{f,t}\) enter **only the reward**, not the policy observation; floor-average CO\(_2\) and floor-average temperatures are used as transferable IAQ/thermal state features.
