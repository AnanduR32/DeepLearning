# Unit III — Calculus: Limits, Derivatives, Multivariable Optimization & Integration
### 25MAT532A — Computational Linear Algebra

This unit develops mathematical analysis and calculus — the language of **continuous change, rates, approximation, and optimization**. In computational linear algebra and machine learning, calculus provides the theoretical and algorithmic engine: cost/loss function minimization $\to$ gradients and Jacobians $\to$ Hessians and curvature $\to$ multivariable chain rule (backpropagation) $\to$ convexity and global optimality $\to$ constrained optimization via Lagrange multipliers $\to$ definite and indefinite integration $\to$ inner product spaces of continuous functions ($L^2$ spaces).

---

## Table of Contents
1. [Numerical Sets, Completeness & Real Functions](#1-numerical-sets-completeness--real-functions)
2. [Limits, Continuity & Asymptotics](#2-limits-continuity--asymptotics)
3. [Derivatives and Single-Variable Differentiability](#3-derivatives-and-single-variable-differentiability)
4. [The Chain Rule & Algorithmic Differentiation](#4-the-chain-rule--algorithmic-differentiation)
5. [The Mean Value Theorem (MVT) & Taylor's Theorem](#5-the-mean-value-theorem-mvt--taylors-theorem)
6. [Convex Sets and Convex Functions](#6-convex-sets-and-convex-functions)
7. [Single-Variable Extrema & The First Derivative Test](#7-single-variable-extrema--the-first-derivative-test)
8. [Single-Variable Extrema & The Second Derivative Test](#8-single-variable-extrema--the-second-derivative-test)
9. [Multivariate Functions, Partial Derivatives & Gradients](#9-multivariate-functions-partial-derivatives--gradients)
10. [Multivariate Differentiability, Jacobians & Hessians](#10-multivariate-differentiability-jacobians--hessians)
11. [Multivariate Optimization, Saddle Points & Lagrange Multipliers](#11-multivariate-optimization-saddle-points--lagrange-multipliers)
12. [The Gradient Descent Optimization Algorithm](#12-the-gradient-descent-optimization-algorithm)
13. [Integration Basics: Riemann Sums & The Fundamental Theorem](#13-integration-basics-riemann-sums--the-fundamental-theorem)
14. [Techniques of Integration & Function Space Inner Products](#14-techniques-of-integration--function-space-inner-products)
15. [Comprehensive Solved Exam-Style Problems](#15-comprehensive-solved-exam-style-problems)
16. [Unit III Summary & Formula Cheat Sheet](#16-unit-iii-summary--formula-cheat-sheet)

---

## 1. Numerical Sets, Completeness & Real Functions

### 1.1 Number Systems and the Completeness Axiom

Calculus is built upon the hierarchy of numerical sets:
$$\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$$

```
The Real Number Hierarchy:
   Natural Numbers (N) = {1, 2, 3, ...}
   Integers (Z)        = {..., -2, -1, 0, 1, 2, ...}
   Rationals (Q)       = {p / q : p, q in Z, q != 0}  (Contains "holes", e.g. sqrt(2))
   Reals (R)           = Q U Irrationals              (Complete: NO holes/gaps!)
   Complex (C)         = {a + b*i : a, b in R, i^2 = -1}
```

#### The Completeness Axiom (Least Upper Bound Property)
The set of rational numbers $\mathbb{Q}$ is incomplete: the bounded set $\{q \in \mathbb{Q} \mid q^2 < 2\}$ has upper bounds in $\mathbb{Q}$ (e.g., $1.5, 2$), but **no rational least upper bound** (since $\sqrt{2} \notin \mathbb{Q}$).

**Completeness Axiom:** Every non-empty subset $S \subset \mathbb{R}$ that is bounded above possesses a **supremum** (least upper bound) in $\mathbb{R}$:
$$\sup(S) \in \mathbb{R}$$
This completeness property guarantees that limits, continuous intermediate values (Intermediate Value Theorem), and extreme values on closed intervals (Extreme Value Theorem) actually exist as real numbers.

---

## 2. Limits, Continuity & Asymptotics

### 2.1 Formal $\epsilon$-$\delta$ Limit Definition

The statement $\lim_{x\to c} f(x) = L$ means: as $x$ gets arbitrarily close to $c$ ($x \neq c$), $f(x)$ gets arbitrarily close to $L$.

```
Visualizing the Epsilon-Delta Limit Definition:
       y ^
         |
   L + e + - - - - - - - - - - +-------------+
         |                     |             |
       L +---------------------+-- f(c) -----+  |f(x) - L| < epsilon
         |                     |             |
   L - e + - - - - - - - - - - +-------------+
         |                     |             |
       0 +---------------------+------+------+--------> x
                             c - d    c    c + d
                               |<- 2*delta ->|
```

#### Formal Definition
$$\forall \epsilon > 0, \; \exists \delta > 0 \text{ such that } 0 < |x - c| < \delta \implies |f(x) - L| < \epsilon$$

#### Continuity at a Point
A function $f$ is **continuous at $x = c$** if:
1. $f(c)$ is defined.
2. $\lim_{x\to c} f(x)$ exists ($\lim_{x\to c^-} f(x) = \lim_{x\to c^+} f(x)$).
3. $\lim_{x\to c} f(x) = f(c)$.

#### Classification of Discontinuities
- **Removable Discontinuity:** $\lim_{x\to c} f(x)$ exists, but $\lim_{x\to c} f(x) \neq f(c)$ (or $f(c)$ is undefined). E.g., $f(x) = \frac{x^2 - 4}{x - 2}$ at $x=2$.
- **Jump Discontinuity:** Left and right limits exist but are unequal: $\lim_{x\to c^-} f(x) \neq \lim_{x\to c^+} f(x)$. E.g., $\text{sgn}(x)$ at $x=0$.
- **Essential / Infinite Discontinuity:** At least one one-sided limit does not exist or blows up to $\pm \infty$. E.g., $f(x) = \frac{1}{x}$ or $\sin(1/x)$ at $x=0$.

---

## 3. Derivatives and Single-Variable Differentiability

### 3.1 The Limit of Difference Quotients

The derivative $f'(x)$ represents the **instantaneous rate of change** and the **slope of the tangent line** to $y = f(x)$ at $(x, f(x))$:
$$f'(x) = \frac{df}{dx} = \lim_{h\to 0} \frac{f(x + h) - f(x)}{h}$$

```
Secant Line Approaching Tangent Line as h -> 0:
       y ^
         |                                 * (x+h, f(x+h))
         |                               / |
         |                   Secant    /   |
         |                   Slope   /     | f(x+h) - f(x)
         |               * --------+       |
         |             / | (Tangent)       |
         |           /   |                 |
         |   (x,f(x))    |                 |
         +---------------+-----------------+--------> x
                         x                x+h
                         |<------ h ------>|
```

#### Fundamental Differentiation Rules
- **Power Rule:** $\frac{d}{dx}[x^n] = n x^{n-1}$
- **Sum/Difference Rule:** $(f \pm g)'(x) = f'(x) \pm g'(x)$
- **Product Rule:** $(f g)'(x) = f'(x) g(x) + f(x) g'(x)$
- **Quotient Rule:** $\left( \frac{f}{g} \right)'(x) = \frac{f'(x) g(x) - f(x) g'(x)}{[g(x)]^2}$

#### Proof: Differentiability Implies Continuity
**Theorem:** If $f$ is differentiable at $x = c$, then $f$ is continuous at $x = c$.
*Proof:* To show continuity, we must show $\lim_{x\to c} [f(x) - f(c)] = 0$.
For $x \neq c$:
$$f(x) - f(c) = \frac{f(x) - f(c)}{x - c} \cdot (x - c)$$
Taking the limit as $x \to c$:
$$\lim_{x\to c} [f(x) - f(c)] = \lim_{x\to c} \left( \frac{f(x) - f(c)}{x - c} \right) \cdot \lim_{x\to c} (x - c) = f'(c) \cdot 0 = 0$$
Thus $\lim_{x\to c} f(x) = f(c)$, proving continuity. $\blacksquare$

*Nuance:* The converse is **false**. $f(x) = |x|$ is continuous everywhere, but not differentiable at $x=0$ because the left slope is $-1$ and the right slope is $+1$.

---

## 4. The Chain Rule & Algorithmic Differentiation

### 4.1 Theory & Machine Learning Backpropagation

For composite function $h(x) = (f \circ g)(x) = f(g(x))$:
$$h'(x) = f'(g(x)) \cdot g'(x) \iff \frac{dh}{dx} = \frac{dh}{du} \cdot \frac{du}{dx} \quad (\text{where } u = g(x))$$

```
Computational Graph of Chain Rule:
       x  ---> [ Function g ] ---> u = g(x) ---> [ Function f ] ---> y = f(u)
                  g'(x)                             f'(u)
       <----------------- (dy/dx = f'(u) * g'(x)) -----------------
```

#### Application to Neural Network Backpropagation
A deep neural network computes $y = f_L(f_{L-1}(\cdots f_1(x)))$. To compute the gradient of loss $\mathcal{L}$ with respect to intermediate weights $W_k$, the chain rule multiplies Jacobians backward:
$$\frac{\partial \mathcal{L}}{\partial W_k} = \frac{\partial \mathcal{L}}{\partial a_L} \frac{\partial a_L}{\partial a_{L-1}} \cdots \frac{\partial a_{k+1}}{\partial a_k} \frac{\partial a_k}{\partial W_k}$$
Automatic Differentiation (AD in PyTorch/TensorFlow) is the exact programmatic implementation of the multivariable chain rule.

---

## 5. The Mean Value Theorem (MVT) & Taylor's Theorem

### 5.1 Rolle's Theorem & Lagrange Mean Value Theorem

#### Rolle's Theorem
If $f$ is continuous on $[a, b]$, differentiable on $(a, b)$, and $f(a) = f(b)$, then there exists at least one $c \in (a, b)$ such that:
$$f'(c) = 0$$

#### Lagrange Mean Value Theorem (MVT)
If $f$ is continuous on $[a, b]$ and differentiable on $(a, b)$, then there exists $c \in (a, b)$ such that:
$$f'(c) = \frac{f(b) - f(a)}{b - a}$$

```
Geometric Interpretation of the Mean Value Theorem:
       y ^                                          * (b, f(b))
         |                                        /
         |                             Tangent  /   Secant Line
         |                             Slope  /     Slope = (f(b)-f(a))/(b-a)
         |                       * - - - - -/
         |                     / |        /
         |           (a, f(a))*  |      /
         |                    |  |    /
         +--------------------+--+---+------------------------> x
                              a  c   b
```

#### Proof of MVT via Rolle's Theorem
*Proof:* Define the auxiliary function $\phi(x)$ representing the vertical distance between $f(x)$ and the secant line:
$$\phi(x) = f(x) - f(a) - \frac{f(b) - f(a)}{b - a}(x - a)$$
Note that $\phi(a) = 0$ and $\phi(b) = f(b) - f(a) - [f(b) - f(a)] = 0$.
Since $\phi(x)$ is continuous on $[a, b]$ and differentiable on $(a, b)$ with $\phi(a) = \phi(b) = 0$, Rolle's Theorem guarantees $\exists c \in (a, b)$ such that $\phi'(c) = 0$:
$$\phi'(c) = f'(c) - \frac{f(b) - f(a)}{b - a} = 0 \implies f'(c) = \frac{f(b) - f(a)}{b - a} \quad \blacksquare$$

#### Taylor's Theorem with Lagrange Remainder
For an $(n+1)$-times differentiable function $f$:
$$f(x) = f(a) + f'(a)(x - a) + \frac{f''(a)}{2!}(x - a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x - a)^n + R_n(x)$$
where the **Lagrange Remainder** is $R_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x - a)^{n+1}$ for some $\xi$ between $a$ and $x$.

---

## 6. Convex Sets and Convex Functions

### 6.1 Definitions, Characterizations & Jensen's Inequality

A set $C \subset \mathbb{R}^n$ is **convex** if for all $x, y \in C$ and $\lambda \in [0, 1]$:
$$\lambda x + (1 - \lambda) y \in C$$

A function $f: C \to \mathbb{R}$ is **convex** if its epigraph is a convex set, meaning for all $x, y \in C$ and $\lambda \in [0, 1]$:
$$f(\lambda x + (1 - \lambda) y) \le \lambda f(x) + (1 - \lambda) f(y)$$

```
Convex vs Non-Convex Functions:
       Convex Function:                       Non-Convex Function:
       y ^                                    y ^
         |      Chord Above Graph               |        Chord Crosses Graph
         |     *---------------+                |     *-------\------+
         |    / \             /                 |    / \       \    / \
         |   /   \___________/                  |   /   \_______*--/   \
         +-------------------------> x          +-------------------------> x
         Every Local Min IS Global Min!         Trapped in Suboptimal Local Minima!
```

#### Differential Characterizations of Convexity
1. **First-Order Condition (Tangent Hyperplane Underestimator):**
   $$f(y) \ge f(x) + \nabla f(x)^T (y - x) \quad \forall x, y$$
2. **Second-Order Condition (Positive Semi-Definite Curvature):**
   - Single-variable: $f''(x) \ge 0$ for all $x$.
   - Multivariable: The Hessian matrix $H(x) = \nabla^2 f(x) \succeq 0$ (positive semi-definite) for all $x$.

#### Jensen's Inequality
For any convex function $f$ and random variable $X$:
$$\mathbb{E}[f(X)] \ge f(\mathbb{E}[X])$$

---

## 7. Single-Variable Extrema & The First Derivative Test

### 7.1 Critical Points & Local vs Global Extrema

- **Critical Point:** A point $c$ in the domain of $f$ where $f'(c) = 0$ or $f'(c)$ is undefined.
- **Fermat's Theorem on Stationary Points:** If $f$ has a local extremum at an interior point $c$ and is differentiable at $c$, then $f'(c) = 0$.

#### The First Derivative Test Algorithm
Evaluate the sign of $f'(x)$ on intervals partitioned by critical points:
1. **Local Maximum:** $f'(x)$ changes sign from **positive ($+$) to negative ($-$)** as $x$ passes through $c$.
2. **Local Minimum:** $f'(x)$ changes sign from **negative ($-$) to positive ($+$)** as $x$ passes through $c$.
3. **No Extremum (Inflection / Terrace Point):** $f'(x)$ maintains the same sign on both sides of $c$ (e.g., $f(x) = x^3$ at $x=0$).

---

## 8. Single-Variable Extrema & The Second Derivative Test

### 8.1 Curvature-Based Classification

At a stationary point where $f'(c) = 0$:
1. **$f''(c) > 0$:** The graph is strictly concave upward (convex) $\implies$ **Local Minimum**.
2. **$f''(c) < 0$:** The graph is strictly concave downward $\implies$ **Local Maximum**.
3. **$f''(c) = 0$:** **Inconclusive**! Fall back to the First Derivative Test.

```
Why f''(c) = 0 is Inconclusive:
   f(x) = x^4  ---> f'(0)=0, f''(0)=0 ---> Has a strict LOCAL MINIMUM at x=0.
   f(x) = -x^4 ---> f'(0)=0, f''(0)=0 ---> Has a strict LOCAL MAXIMUM at x=0.
   f(x) = x^3  ---> f'(0)=0, f''(0)=0 ---> Has an INFLECTION POINT at x=0.
```

---

## 9. Multivariate Functions, Partial Derivatives & Gradients

### 9.1 Functions of Several Variables & Directional Derivatives

For $f: \mathbb{R}^n \to \mathbb{R}$ (e.g., $f(x, y)$), the **partial derivative** w.r.t. $x_i$ measures the rate of change holding all other variables constant:
$$\frac{\partial f}{\partial x_i} = \lim_{h\to 0} \frac{f(x_1, \dots, x_i + h, \dots, x_n) - f(x_1, \dots, x_n)}{h}$$

#### The Gradient Vector ($\nabla f$)
$$\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix} \in \mathbb{R}^n$$

#### Directional Derivative ($D_u f$)
For any unit direction vector $u \in \mathbb{R}^n$ ($\|u\|_2 = 1$):
$$D_u f(x) = \lim_{h\to 0} \frac{f(x + h u) - f(x)}{h} = \nabla f(x) \cdot u = \|\nabla f(x)\|_2 \cos\theta$$

```
Contour Map, Level Curves and Gradient Direction:
       y ^
         |              Level Curves f(x,y) = k
         |             /  /  /
         |            (  (  (    ^ \nabla f (Perpendicular to Level Curve!)
         |             \  \  \  /   Points in Direction of STEEPEST ASCENT
         |              \  \  * 
         |               \  \  \
         +---------------------------------> x
```

#### Proof: Gradient Points in Direction of Steepest Ascent
$$\max_{\|u\|_2 = 1} D_u f(x) = \max_{\|u\|_2 = 1} (\nabla f(x) \cdot u) = \|\nabla f(x)\|_2$$
By the Cauchy–Schwarz inequality, the maximum dot product occurs when $\cos\theta = 1 \iff u = \frac{\nabla f(x)}{\|\nabla f(x)\|_2}$. Thus, the gradient vector points in the direction of **maximum rate of increase**, with magnitude equal to that maximum rate. $\blacksquare$

---

## 10. Multivariate Differentiability, Jacobians & Hessians

### 10.1 Formal Differentiability & The $C^1$ Criterion

A multivariable function $f: \mathbb{R}^n \to \mathbb{R}$ is **differentiable at $x_0$** if there exists a linear approximation whose error decays faster than $\|h\|$:
$$\lim_{h\to 0} \frac{f(x_0 + h) - f(x_0) - \nabla f(x_0)^T h}{\|h\|_2} = 0$$

*Subtlety / Counterexample:* The existence of all partial derivatives $\frac{\partial f}{\partial x_i}$ at a point **does NOT guarantee differentiability or even continuity**!
Consider $f(x, y) = \frac{x y}{x^2 + y^2}$ for $(x, y) \neq (0, 0)$ and $f(0, 0) = 0$.
Both $\frac{\partial f}{\partial x}(0, 0) = 0$ and $\frac{\partial f}{\partial y}(0, 0) = 0$, but along the line $y = x$, $f(x, x) = \frac{x^2}{2x^2} = \frac{1}{2} \neq 0$. $f$ is not even continuous at the origin!

**Sufficient Condition ($C^1$ Theorem):** If all partial derivatives $\frac{\partial f}{\partial x_i}$ exist and are **continuous** in a neighborhood of $x_0$, then $f$ is differentiable at $x_0$.

#### The Jacobian Matrix
For a vector-valued function $F: \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian matrix** $J_F(x) \in \mathbb{R}^{m\times n}$ contains all first-order partial derivatives:
$$J_F(x) = \begin{pmatrix} \nabla F_1(x)^T \\ \nabla F_2(x)^T \\ \vdots \\ \nabla F_m(x)^T \end{pmatrix} = \begin{pmatrix} \frac{\partial F_1}{\partial x_1} & \cdots & \frac{\partial F_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial F_m}{\partial x_1} & \cdots & \frac{\partial F_m}{\partial x_n} \end{pmatrix}$$

#### The Hessian Matrix ($\nabla^2 f$) & Multivariable Taylor Series
For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian matrix** $H \in \mathbb{R}^{n\times n}$ is the symmetric matrix of second-order partial derivatives (by Clairaut's Theorem):
$$H(x) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$

Second-order multivariable Taylor expansion around $x_0$:
$$f(x_0 + h) \approx f(x_0) + \nabla f(x_0)^T h + \frac{1}{2} h^T H(x_0) h$$

---

## 11. Multivariate Optimization, Saddle Points & Lagrange Multipliers

### 11.1 Critical Points & Hessian Definiteness Test

At a stationary point $x^*$ where $\nabla f(x^*) = 0$:
1. **Local Minimum:** $H(x^*) \succ 0$ (Hessian is **Positive Definite** $\iff$ all eigenvalues $\lambda_i > 0$).
2. **Local Maximum:** $H(x^*) \prec 0$ (Hessian is **Negative Definite** $\iff$ all eigenvalues $\lambda_i < 0$).
3. **Saddle Point:** $H(x^*)$ is **Indefinite** (has both positive and negative eigenvalues).
4. **Inconclusive:** $H(x^*)$ is semi-definite (at least one $\lambda_i = 0$).

```
2D Curvature Types:
   Local Minimum (Bowl)       Local Maximum (Dome)        Saddle Point (Pringles Chip)
          z ^                        z ^                         z ^
            |                          |                           |      / (Up along x)
          \___/                      /~~~\                         |    /
         (H > 0)                    (H < 0)                        +--*--------> x
                                                                     \   \
                                                                       \   \ (Down along y)
```

#### 2D Discriminant Test ($D = f_{xx} f_{yy} - f_{xy}^2$)
For $f(x, y)$ at stationary point $(x_0, y_0)$:
- $D = \det(H) = f_{xx} f_{yy} - (f_{xy})^2$.
- If $D > 0$ and $f_{xx} > 0 \implies$ **Local Minimum**.
- If $D > 0$ and $f_{xx} < 0 \implies$ **Local Maximum**.
- If $D < 0 \implies$ **Saddle Point**.
- If $D = 0 \implies$ **Inconclusive**.

#### Constrained Optimization via Lagrange Multipliers
To optimize $f(x)$ subject to equality constraint $g(x) = 0$:
At the constrained optimum, the contour lines of $f$ must be tangent to the constraint surface $g = 0$, meaning their gradient vectors are parallel:
$$\nabla f(x^*) = \lambda \nabla g(x^*), \quad g(x^*) = 0$$
where $\lambda \in \mathbb{R}$ is the **Lagrange multiplier**.

---

## 12. The Gradient Descent Optimization Algorithm

### 12.1 Unconstrained First-Order Optimization

Gradient descent is the foundational iterative solver for training machine learning models:
$$x^{(k+1)} = x^{(k)} - \eta \nabla f(x^{(k)})$$
where $\eta > 0$ is the **learning rate / step size**.

- If $\eta$ is too small: Extremely slow convergence.
- If $\eta$ is too large: Oscillations, overshoot, or divergence.
- For a convex quadratic $f(x) = \frac{1}{2} x^T A x - b^T x$ with $A \succ 0$, convergence is guaranteed if $0 < \eta < \frac{2}{\lambda_{\max}(A)}$.

---

## 13. Integration Basics: Riemann Sums & The Fundamental Theorem

### 13.1 Definite Integrals and The Fundamental Theorem of Calculus

The definite integral is the limit of Riemann sums as partition width $\Delta x_i \to 0$:
$$\int_a^b f(x) dx = \lim_{n\to\infty} \sum_{i=1}^n f(x_i^*) \Delta x_i$$

```
Riemann Sum Approximating Area Under Curve:
       y ^
         |              /~~~\
         |            / |   | \
         |          /|  |   |  |\
         |         | |  |   |  | |  Area = Sum f(x_i*) * Delta x
         +---------+-+--+---+--+-+----------> x
                   a = x_0 ... x_n = b
```

#### Fundamental Theorem of Calculus (FTC)
- **Part 1 (Derivative of Accumulation Function):**
  $$\frac{d}{dx} \left[ \int_a^x f(t) dt \right] = f(x)$$
- **Part 2 (Evaluation Theorem):**
  $$\int_a^b f(x) dx = F(b) - F(a), \quad \text{where } F'(x) = f(x)$$

---

## 14. Techniques of Integration & Function Space Inner Products

### 14.1 Key Methods & Inner Products in $C[a, b]$

1. **Integration by Substitution ($u$-sub):**
   $$\int f(g(x)) g'(x) dx = \int f(u) du, \quad u = g(x)$$
2. **Integration by Parts:**
   $$\int u \, dv = u v - \int v \, du$$

#### Numerical Quadrature
- **Trapezoidal Rule:** $\int_a^b f(x) dx \approx \frac{\Delta x}{2} [f(x_0) + 2f(x_1) + \cdots + 2f(x_{n-1}) + f(x_n)]$.
- **Simpson's Rule:** $\int_a^b f(x) dx \approx \frac{\Delta x}{3} [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + f(x_n)]$.

#### The Bridge: Inner Product on Function Spaces ($L^2[a, b]$)
The set of continuous real-valued functions $C[a, b]$ forms an infinite-dimensional vector space under the $L^2$ inner product:
$$\langle f, g \rangle = \int_a^b f(x) g(x) dx$$
- Induced $L^2$ Norm: $\|f\|_{L^2} = \sqrt{\int_a^b [f(x)]^2 dx}$.
- Orthogonality: $f \perp g \iff \int_a^b f(x) g(x) dx = 0$. (Basis of Fourier series: $\int_{-\pi}^\pi \sin(nx)\cos(mx)dx = 0$).

---

## 15. Comprehensive Solved Exam-Style Problems

### Problem 1: Multivariable Extrema and Saddle Point Classification
**Statement:** Find and classify all critical points of $f(x, y) = 2x^3 + x y^2 + 5x^2 + y^2$.

**Step-by-step Solution:**
1. **Find Gradient $\nabla f(x, y) = 0$:**
   $$f_x = 6x^2 + y^2 + 10x = 0$$
   $$f_y = 2xy + 2y = 2y(x + 1) = 0$$
2. **Solve the System:**
   From $f_y = 0 \implies y = 0$ or $x = -1$.
   - **Case 1: $y = 0$:**
     $$6x^2 + 10x = 0 \implies 2x(3x + 5) = 0 \implies x = 0 \text{ or } x = -5/3$$
     Critical points: $(0, 0)$ and $(-5/3, 0)$.
   - **Case 2: $x = -1$:**
     $$6(-1)^2 + y^2 + 10(-1) = 0 \implies 6 + y^2 - 10 = 0 \implies y^2 = 4 \implies y = \pm 2$$
     Critical points: $(-1, 2)$ and $(-1, -2)$.
3. **Compute Second Partials and Discriminant $D = f_{xx} f_{yy} - f_{xy}^2$:**
   $$f_{xx} = 12x + 10, \quad f_{yy} = 2x + 2, \quad f_{xy} = 2y$$
   $$D(x, y) = (12x + 10)(2x + 2) - 4y^2$$
4. **Classify Each Critical Point:**
   - **At $(0, 0)$:** $D = (10)(2) - 0 = 20 > 0$ and $f_{xx} = 10 > 0 \implies$ **Local Minimum** ($f(0,0)=0$).
   - **At $(-5/3, 0)$:** $D = (-10)(-4/3) - 0 = \frac{40}{3} > 0$ and $f_{xx} = -10 < 0 \implies$ **Local Maximum** ($f(-5/3, 0) = 125/27$).
   - **At $(-1, 2)$:** $D = (-2)(0) - 4(4) = -16 < 0 \implies$ **Saddle Point**.
   - **At $(-1, -2)$:** $D = (-2)(0) - 4(4) = -16 < 0 \implies$ **Saddle Point**.

---

### Problem 2: Constrained Optimization via Lagrange Multipliers
**Statement:** Find the maximum and minimum values of $f(x, y) = x^2 + 2y^2$ on the unit circle $g(x, y) = x^2 + y^2 - 1 = 0$.

**Step-by-step Solution:**
1. **Set Up Lagrange Equations $\nabla f = \lambda \nabla g$:**
   $$\nabla f = \begin{pmatrix} 2x \\ 4y \end{pmatrix}, \qquad \lambda \nabla g = \lambda \begin{pmatrix} 2x \\ 2y \end{pmatrix}$$
   $$\begin{aligned}
   2x &= 2\lambda x \implies 2x(1 - \lambda) = 0 \\
   4y &= 2\lambda y \implies 2y(2 - \lambda) = 0 \\
   x^2 + y^2 &= 1
   \end{aligned}$$
2. **Analyze Branching Conditions:**
   - From Eq. 1: $x = 0$ or $\lambda = 1$.
     - If $x = 0$: $x^2 + y^2 = 1 \implies y = \pm 1$. $\lambda = 2$.
       Points: $(0, 1)$ and $(0, -1)$. Value: $f(0, \pm 1) = 0 + 2(1) = \mathbf{2}$ (MAXIMUM).
     - If $\lambda = 1$: From Eq. 2, $2y(2 - 1) = 2y = 0 \implies y = 0$.
       $x^2 + 0 = 1 \implies x = \pm 1$.
       Points: $(1, 0)$ and $(-1, 0)$. Value: $f(\pm 1, 0) = 1 + 0 = \mathbf{1}$ (MINIMUM).

---

## 16. Unit III Summary & Formula Cheat Sheet

| Mathematical Concept | Defining Formula / Key Theorem | Applied Relevance in AI / Engineering |
|---|---|---|
| **Limit ($\epsilon$-$\delta$)** | $\forall \epsilon > 0, \; \exists \delta > 0 : 0 < \vert x - c \vert < \delta \implies \vert f(x) - L \vert < \epsilon$ | Rigorous asymptotic convergence analysis |
| **Derivative** | $f'(x) = \lim_{h\to 0} \frac{f(x+h)-f(x)}{h}$ | Instantaneous sensitivity / slope of loss surface |
| **Chain Rule** | $\frac{dh}{dx} = \frac{dh}{du}\frac{du}{dx}$ | Neural network backpropagation & Automatic Diff |
| **Mean Value Theorem** | $f'(c) = \frac{f(b)-f(a)}{b-a}$ | Error bounds in numerical approximation & Taylor series |
| **Convexity** | $f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda)f(y)$ | Guarantees any local minimum is a global minimum |
| **Gradient Vector** | $\nabla f = (\partial f/\partial x_1, \dots, \partial f/\partial x_n)^T$ | Points in direction of steepest ascent ($\max D_u f$) |
| **Hessian Matrix** | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}, \; H \succ 0 \implies \text{Local Min}$ | Second-order curvature, Newton's optimization method |
| **Lagrange Multipliers** | $\nabla f = \lambda \nabla g, \; g(x) = 0$ | Constrained optimization (SVMs, PCA derivation) |
| **Gradient Descent** | $x^{(k+1)} = x^{(k)} - \eta \nabla f(x^{(k)})$ | Workhorse parameter optimization in Deep Learning |
| **Fundamental Theorem of Calculus** | $\int_a^b f(x)dx = F(b) - F(a)$ | Area accumulation, probability distributions, $L^2$ norms |
