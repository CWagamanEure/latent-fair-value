# Perp-Spot Latent Efficient Price Filtering Notes

## Core idea

A Kalman filter is not a single universal "efficient price finder." It is a **framework for latent-state estimation**. The filter itself is generic, but the meaning of the latent states is **asset-specific** and **horizon-specific**.

That means:

* There is no one universal "Kalman filter for efficient price."
* The part that changes across assets is the **state definition**, **dynamics**, **measurements**, **inputs/covariates**, and **noise assumptions**.
* The filter is the inference engine that combines these pieces.

A good summary line is:

> The filter is generic; the economic meaning of the states is asset-specific.

---

## Efficient price estimation vs short-horizon forecasting

Two related but distinct goals often get mixed together:

### 1. Efficient/fundamental price estimation

This is the latent "true" price after filtering out:

* bid-ask bounce
* microstructure noise
* stale quotes
* temporary dislocations
* transient impact

### 2. Short-horizon predictive price estimation

This is more like:

* where is the next mid likely to move?
* how do microprice, OFI, queue imbalance, and spread affect the next quote revision?

These are related, but they are not the same thing.

A variable can be highly predictive of the next move without literally being the latent efficient price.

For example:

* queue imbalance may predict the next mid change
* but that may reflect short-run liquidity pressure rather than a change in long-run value

So something like **microprice** may be better treated as:

* a signal about short-run pressure
* or a measurement of a pressure state
* rather than automatically the efficient price itself

---

## Why asset-specific structure matters

Different assets can have different dominant mechanisms:

* some are well described by a denoised midprice
* some have very informative queue imbalance and microprice
* some are driven by cross-venue price discovery
* some need perp/spot basis modeling
* some need ETF/NAV premium-discount modeling
* some are heavily influenced by discrete tick effects, inventory pressure, or auction structure

So the right question is not:

> What is the one correct efficient price model?

The better question is:

> What latent object do I want to estimate for this asset and horizon?

That object could be:

* a smooth efficient price underlying noisy quotes
* a common price across multiple venues
* a fair spot value plus basis component for a perp
* a short-horizon clearing/fair price conditional on order book pressure
* a fundamental value plus transient dislocation

---

## What the filter is really doing

It is fair to say that the filter controls how much each information source affects the latent estimate, but that statement needs to be made carefully.

A more accurate statement is:

> The filter determines how much each information source should move the latent state estimate, given the assumed state dynamics and the assumed noise levels.

At each step, the filter balances:

* what the latent state was expected to do on its own
* what the new observations are saying

So:

* if a measurement is noisy, the filter trusts it less
* if a measurement is precise, the filter trusts it more
* if the state is assumed stable, the filter updates more cautiously
* if the state is assumed volatile, the filter updates more aggressively

This is more subtle than ordinary regression weights because the influence is dynamic and depends on current uncertainty and model structure.

---

## Three different roles that variables can play

One of the most important modeling questions is:

> For each variable, is it a measurement, a latent state component, or an exogenous driver/covariate?

### 1. Measurements

Measurements are noisy observations of the latent state.

Examples:

* observed spot price
* observed perp price
* trade price
* mark price
* oracle price
* ETF price

These enter the **observation equation**:

[
z_t = H x_t + v_t
]

### 2. Latent state components

Sometimes what looks like a "feature" is actually something that should be estimated as its own hidden state.

Examples:

* efficient/common price
* equilibrium basis
* temporary dislocation
* short-run pressure
* venue-specific distortion

These live inside the hidden state vector.

### 3. Exogenous drivers / covariates

A covariate is an observed variable that helps explain how the latent state moves, even though it is not itself the latent state.

Examples:

* funding rate
* open interest change
* OFI
* queue imbalance
* spread
* depth imbalance

These often enter the **state transition equation**:

[
x_t = A x_{t-1} + B u_t + w_t
]

where (u_t) is a vector of external inputs/covariates.

---

## What a covariate means in plain English

A covariate is just an **observed explanatory input**.

It is not the hidden state itself, and it is often not a direct noisy observation of the state either. Instead, it helps explain why the state changes.

In plain English:

> A covariate is some observed variable I include because it contains explanatory information.

Other words used in different contexts are:

* predictor
* regressor
* explanatory variable
* input

### Example analogy

Suppose you are estimating a car's hidden true speed.

* hidden state: true speed
* measurement: noisy speedometer reading
* covariate: road slope, wind, throttle position

The slope is not the speed, and it is not a noisy reading of speed, but it helps explain why speed changes.

That is what a covariate is.

### In a perp-spot model

Examples of covariates:

* funding rate
* open interest change
* OFI
* queue imbalance
* spread

These may be used to drive the hidden basis or temporary dislocation states.

One subtlety: the same variable can play different roles in different models. For example, microprice could be modeled as:

* a covariate driving short-run state changes
* a noisy measurement of a latent pressure state
* a derived signal used outside the filter

So "covariate" is not an intrinsic property of the variable. It depends on how you place it in the model.

---

## A useful mental model: layers of price formation

You can think of financial price filtering as layered:

1. **Latent efficient/common price**
2. **Medium-speed structural deviations**
3. **Very short-lived microstructure distortions**
4. **Noisy observed prices and signals**

Different assets differ mainly in which layers matter most.

---

## A practical hierarchy of model types

Instead of searching for one universal model, it is useful to think in families:

### Model A: baseline denoising

* latent efficient price random walk
* mid/trade as noisy measurements

### Model B: efficient price + transient microstructure pressure

* add a mean-reverting short-run state
* use imbalance, OFI, microprice, spread

### Model C: multi-venue common price model

* one common efficient price
* venue-specific temporary distortions

### Model D: asset-structure-specific model

* perp basis state
* ETF premium/discount state
* funding-linked state
* open/close or auction effects if relevant

These can be compared out of sample by:

* next-return forecast quality
* quote revision forecast quality
* execution benchmark improvement
* innovation diagnostics
* filtered residual autocorrelation

---

## Perp-spot latent efficient price filter: core decomposition

For a perp-spot system, a natural state decomposition is:

* (m_t): efficient spot price
* (b_t): equilibrium perp basis
* (d_t): temporary perp dislocation around that basis

Then the perp price is interpreted as:

[
\text{observed perp} = \text{efficient spot} + \text{structural basis} + \text{temporary deviation} + \text{measurement noise}
]

And the spot price is interpreted as:

[
\text{observed spot} = \text{efficient spot} + \text{measurement noise}
]

This is a coherent state-space story, not just a bag of indicators.

---

## Important modeling refinement

A key refinement is that many things that feel like "features" should not all be treated as direct measurements.

A clean division is:

### Latent states

* efficient/common spot price
* equilibrium perp basis
* temporary dislocation

### Measurements

* observed spot price
* observed perp price
* possibly oracle or mark price if available

### State drivers / covariates

* funding, premium history, open interest for basis evolution
* microstructure signals such as OFI, queue imbalance, spread, and maybe microprice residuals for temporary dislocation

This is cleaner than treating funding or microstructure variables as direct observations of price.

---

## The clean linear Gaussian state-space model

Let the hidden state vector be:

[
x_t =
\begin{bmatrix}
m_t \
b_t \
d_t
\end{bmatrix}
]

where:

* (m_t) = efficient spot price
* (b_t) = equilibrium perp basis
* (d_t) = temporary perp dislocation

Let the observed price vector be:

[
y_t =
\begin{bmatrix}
p^{spot}_t \
p^{perp}_t
\end{bmatrix}
]

### Observation equation

[
y_t = H x_t + v_t
]

with

[
H =
\begin{bmatrix}
1 & 0 & 0 \
1 & 1 & 1
\end{bmatrix}
]

So explicitly:

[
p^{spot}_t = m_t + v^{spot}_t
]

[
p^{perp}_t = m_t + b_t + d_t + v^{perp}_t
]

This means:

* spot observes the efficient spot price, plus noise
* perp observes efficient spot plus structural basis plus temporary deviation, plus noise

---

## State transition equation

Let the state evolve as:

[
x_t = A x_{t-1} + B u_t + w_t
]

where (u_t) is a vector of observed covariates.

A natural covariate vector is:

[
u_t =
\begin{bmatrix}
f_t \
\Delta OI_t \
q_t \
ofi_t \
s_t
\end{bmatrix}
]

where for example:

* (f_t) = funding rate
* (\Delta OI_t) = change in open interest
* (q_t) = queue imbalance
* (ofi_t) = order flow imbalance
* (s_t) = relative spread

A simple transition matrix is:

[
A =
\begin{bmatrix}
1 & 0 & 0 \
0 & \phi_b & 0 \
0 & 0 & \phi_d
\end{bmatrix}
]

A simple covariate loading matrix is:

[
B =
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 \
\beta_f & \beta_{oi} & 0 & 0 & 0 \
0 & 0 & \gamma_q & \gamma_{ofi} & \gamma_s
\end{bmatrix}
]

This implies the following component equations:

### Efficient spot price

[
m_t = m_{t-1} + w^m_t
]

A random walk for the common efficient spot price.

### Basis dynamics

[
b_t = \phi_b b_{t-1} + \beta_f f_t + \beta_{oi}\Delta OI_t + w^b_t
]

This says the basis is:

* persistent through (\phi_b b_{t-1})
* affected by funding and open interest
* subject to additional random shocks

If (|\phi_b| < 1), basis mean-reverts.
If (\phi_b \approx 1), basis behaves more like a slow random walk.

### Temporary dislocation dynamics

[
d_t = \phi_d d_{t-1} + \gamma_q q_t + \gamma_{ofi} ofi_t + \gamma_s s_t + w^d_t
]

This says dislocation is:

* somewhat persistent but mean-reverting
* driven by microstructure pressure
* subject to additional random shocks

Usually you would want:

* (|\phi_d| < 1)
* and typically (|\phi_d|) smaller than (|\phi_b|)

because temporary dislocation should decay faster than structural basis.

---

## Expanded price formulas

Plugging the latent states into the price equations gives:

[
p^{spot}_t = m_t + v^{spot}_t
]

[
p^{perp}_t = m_t + b_t + d_t + v^{perp}_t
]

with

[
b_t = \phi_b b_{t-1} + \beta_f f_t + \beta_{oi}\Delta OI_t + w^b_t
]

[
d_t = \phi_d d_{t-1} + \gamma_q q_t + \gamma_{ofi} ofi_t + \gamma_s s_t + w^d_t
]

In words:

> the perp price equals efficient spot price plus a structural basis driven by funding and open interest plus a temporary deviation driven by microstructure variables

---

## A nicer version: basis reversion toward a factor-driven target

Sometimes it is conceptually cleaner to say that funding and open interest define a **target basis**, and the actual basis moves toward that target.

Define:

[
\theta_t = \beta_f f_t + \beta_{oi}\Delta OI_t
]

Then let basis revert toward that target:

[
b_t = (1-\kappa_b)b_{t-1} + \kappa_b \theta_t + w^b_t
]

Expanding:

[
b_t = (1-\kappa_b)b_{t-1} + \kappa_b\beta_f f_t + \kappa_b\beta_{oi}\Delta OI_t + w^b_t
]

This says:

* funding and open interest define where basis "should" be
* actual basis only moves there gradually

This is often a cleaner economic story than using a plain AR(1) with direct factor loadings.

---

## Spread-form version

If your main interest is the perp-spot gap rather than the full common-price system, it can be simpler to model the spread directly.

Define the observed gap:

[
g_t = p^{perp}_t - p^{spot}_t
]

Then:

[
g_t = b_t + d_t + \epsilon_t
]

with

[
b_t = \phi_b b_{t-1} + \beta_f f_t + \beta_{oi}\Delta OI_t + w^b_t
]

[
d_t = \phi_d d_{t-1} + \gamma_q q_t + \gamma_{ofi} ofi_t + \gamma_s s_t + w^d_t
]

This is simpler if your main focus is decomposing the basis, though it gives up some of the direct common efficient price interpretation unless (m_t) is modeled elsewhere.

---

## Why the separation between basis and temporary dislocation matters

A major risk is **confounding** basis and temporary dislocation.

If both states can explain the same observed perp-spot gap, then the filter may not know whether a move should be assigned to:

* slow structural basis
* or fast temporary dislocation

So the states need to be economically and statistically distinguishable.

Typical identification structure:

* basis moves more slowly than dislocation
* funding and open interest load mainly onto basis
* microstructure variables load mainly onto dislocation
* dislocation mean-reverts faster than basis
* process noise and persistence differ across the two states

That separation is what helps the model remain interpretable.

---

## Subtle point about where the temporary deviation lives

In the simplest model, the temporary deviation mostly lives on the perp side:

[
p^{spot}_t = m_t + v^{spot}_t
]

[
p^{perp}_t = m_t + b_t + d_t + v^{perp}_t
]

This makes sense if spot is treated as the cleaner anchor and perp contains the basis plus temporary distortion.

But if spot itself also has meaningful microstructure distortions, you can extend the model:

[
p^{spot}_t = m_t + d^{spot}_t + \varepsilon^{spot}_t
]

[
p^{perp}_t = m_t + b_t + d^{perp}_t + \varepsilon^{perp}_t
]

where:

* (d^{spot}_t) = temporary spot-side distortion
* (d^{perp}_t) = temporary perp-side distortion

This is more realistic if both venues contribute meaningful microstructure noise.

---

## Where funding belongs

Funding is usually not best interpreted as a direct measurement of efficient price or temporary dislocation.

It is more naturally interpreted as information about the **economic pressure on the basis**.

So funding often belongs in the transition for (b_t), not in the observation equation.

The same idea can apply to:

* open interest
* premium EWMA
* time-to-next-funding effects

These are often basis drivers rather than direct price observations.

---

## Where microprice and microstructure signals belong

There are multiple valid modeling options.

### Option 1: Use them as drivers of temporary dislocation

This is the cleanest first approach.

Examples:

* queue imbalance
* OFI
* relative spread
* top-of-book imbalance
* depth pressure

These enter the dynamics of (d_t).

### Option 2: Introduce a separate latent pressure state

You could define an additional hidden state (s_t) for short-horizon pressure, and let things like microprice or imbalance measure that state.

That is richer, but more complex.

For a first pass, it is very reasonable to use microstructure variables as covariates in the temporary dislocation equation.

---

## Why this is more than ordinary feature weighting

In an ordinary regression you might imagine static coefficients like:

* microprice coefficient = 0.7
* imbalance coefficient = 0.2
* spread coefficient = -0.1

That is a fixed mapping.

A Kalman filter is richer because the influence of new information depends on:

* current uncertainty in the state
* observation noise
* process noise
* correlation across states and measurements
* whether the signal is interpreted as long-run state information or short-run distortion

So the same observed signal can matter more in one moment than another.

In that sense, the filter is deciding:

* how much to trust the prior
* how much to trust the new signal
* which part of the state should get updated
* whether the update should hit efficient price, basis, temporary dislocation, or some mix

---

## Simple conceptual checklist for model design

For any asset and time horizon, ask:

1. What is the horizon?

   * milliseconds
   * seconds
   * minutes
   * longer

2. What is the object of interest?

   * denoised fair price
   * next-move forecast
   * execution benchmark
   * cross-venue price discovery
   * basis decomposition

3. What frictions dominate?

   * bid-ask bounce
   * queue imbalance
   * inventory pressure
   * stale quotes
   * basis/funding effects
   * market-hour mismatch
   * low depth
   * large tick size

4. For each observed variable, is it:

   * a measurement?
   * a covariate / state driver?
   * a hidden state component?

That classification is one of the most important parts of the whole modeling problem.

---

## Bottom-line summary

* Efficient price estimation is not one-size-fits-all.
* That does **not** make Kalman filtering a bad tool.
* It means the **state-space design** must reflect the asset's microstructure and the economic question being asked.
* In a perp-spot setting, a very natural model is:

  * common efficient spot price
  * equilibrium basis
  * temporary dislocation
* Spot and perp prices are the core measurements.
* Funding and open-interest-type variables are natural basis drivers.
* Microstructure variables such as OFI, queue imbalance, and spread are natural temporary-dislocation drivers.
* The clean linear state-space form is:

[
p^{spot}_t = m_t + v^{spot}_t
]

[
p^{perp}_t = m_t + b_t + d_t + v^{perp}_t
]

[
m_t = m_{t-1} + w^m_t
]

[
b_t = \phi_b b_{t-1} + \beta_f f_t + \beta_{oi}\Delta OI_t + w^b_t
]

[
d_t = \phi_d d_{t-1} + \gamma_q q_t + \gamma_{ofi} ofi_t + \gamma_s s_t + w^d_t
]

A very clean conceptual reading of this is:

> observed perp price = efficient spot + structural basis + temporary microstructure-driven deviation + measurement noise

That is exactly the kind of decomposition state-space methods are good at.

