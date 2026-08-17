# Unit II — Eigenvalues, Diagonalization, Iterative Methods & Applications
### 25MAT532A — Computational Linear Algebra

This unit explores the spectral theory of linear transformations: how to find the "natural axes" (eigenvectors) along which a matrix acts by pure scaling (eigenvalues) $\to$ algebraic and geometric multiplicities $\to$ similarity and diagonalization $\to$ the Spectral Theorem and orthogonal diagonalization for symmetric matrices $\to$ Singular Value Decomposition (SVD) for general rectangular matrices $\to$ Principal Component Analysis (PCA) for data dimensionality reduction $\to$ stationary iterative solvers (Jacobi, Gauss–Seidel, and SOR) for massive sparse linear systems $\to$ the Power Method, Inverse Power Method, and Shifted Power Method $\to$ 2D/3D geometric transformations $\to$ the Leslie Demography Model in ecology $\to$ Linear Programming Problems (LPP) and the Simplex method.

---

## Table of Contents
1. [Eigenvalues and Eigenvectors](#1-eigenvalues-and-eigenvectors)
2. [Eigenspaces, Algebraic Multiplicity & Geometric Multiplicity](#2-eigenspaces-algebraic-multiplicity--geometric-multiplicity)
3. [Matrix Diagonalization](#3-matrix-diagonalization)
4. [Orthogonal Diagonalization & The Spectral Theorem](#4-orthogonal-diagonalization--the-spectral-theorem)
5. [Singular Value Decomposition (SVD)](#5-singular-value-decomposition-svd)
6. [Principal Component Analysis (PCA)](#6-principal-component-analysis-pca)
7. [Iterative Methods for Linear Systems (Jacobi, Gauss–Seidel & SOR)](#7-iterative-methods-for-linear-systems-jacobi-gaussseidel--sor)
8. [The Power Method & Eigenvalue Algorithms](#8-the-power-method--eigenvalue-algorithms)
9. [Linear Transformations: Rotation and General Transforms](#9-linear-transformations-rotation-and-general-transforms)
10. [The Leslie Demography Model (Population Dynamics)](#10-the-leslie-demography-model-population-dynamics)
11. [Linear Programming Problems (LPP) & Simplex Foundations](#11-linear-programming-problems-lpp--simplex-foundations)
12. [Comprehensive Solved Exam-Style Problems](#12-comprehensive-solved-exam-style-problems)
13. [Unit II Summary & Formula Cheat Sheet](#13-unit-ii-summary--formula-cheat-sheet)

---

## 1. Eigenvalues and Eigenvectors

### 1.1 Mathematical Definition & Geometric Intuition

For a square matrix $A \in \mathbb{R}^{n\times n}$, a non-zero vector $v \in \mathbb{C}^n$ ($v \neq 0$) is an **eigenvector** of $A$ corresponding to **eigenvalue** $\lambda \in \mathbb{C}$ if:
$$A v = \lambda v$$

Geometrically, the transformation $T_A(x) = Ax$ maps the direction $v$ onto **itself**, scaled by a factor of $\lambda$. There is no rotation or tilting of the vector's axis.

```
Geometric Action of an Eigenvector vs General Vector:
       y ^                                    y ^
         |      Av = lambda * v                 |          Ax
         |     /                                |        /
         |   /                                  |      /
         | /  v                                 |    /
       0 +--------> x                         0 +--------> x
                                                    x
      (Eigenvector: Pure Stretch)           (General Vector: Stretch + Rotate)
```

#### The Characteristic Equation
Rearranging $Av = \lambda v$:
$$(A - \lambda I) v = 0$$
Since $v \neq 0$, the matrix $(A - \lambda I)$ must have a non-trivial null space, which holds if and only if it is singular:
$$p(\lambda) = \det(A - \lambda I) = 0$$
This degree-$n$ polynomial $p(\lambda)$ is the **characteristic polynomial** of $A$. The roots of $p(\lambda) = 0$ are the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$.

#### Fundamental Trace and Determinant Invariants
For any $n\times n$ matrix $A$ with eigenvalues $\lambda_1, \dots, \lambda_n$ (counted with algebraic multiplicity):
1. **Trace Invariant:**
   $$\text{tr}(A) = \sum_{i=1}^n A_{ii} = \sum_{i=1}^n \lambda_i$$
2. **Determinant Invariant:**
   $$\det(A) = \prod_{i=1}^n \lambda_i = \lambda_1 \lambda_2 \cdots \lambda_n$$

#### Proof: Trace and Determinant Invariants
*Proof:* The characteristic polynomial factors over $\mathbb{C}$ as:
$$p(\lambda) = \det(A - \lambda I) = (-1)^n (\lambda - \lambda_1)(\lambda - \lambda_2)\cdots(\lambda - \lambda_n)$$
Evaluating at $\lambda = 0$:
$$p(0) = \det(A) = (-1)^n (-\lambda_1)(-\lambda_2)\cdots(-\lambda_n) = \lambda_1 \lambda_2 \cdots \lambda_n = \det(A)$$
Expanding both the determinant definition and the factored form, the coefficient of $(-\lambda)^{n-1}$ in $\det(A - \lambda I)$ is precisely the sum of the principal diagonal entries $\sum A_{ii} = \text{tr}(A)$, while in the factored polynomial it is $\sum \lambda_i$. Thus, $\text{tr}(A) = \sum \lambda_i$. $\blacksquare$

---

### 1.2 Step-by-Step Worked Example

Find the eigenvalues and eigenvectors of $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$.

1. **Characteristic Equation:**
   $$\det(A - \lambda I) = \begin{vmatrix} 4 - \lambda & 1 \\ 2 & 3 - \lambda \end{vmatrix} = (4 - \lambda)(3 - \lambda) - (1)(2) = \lambda^2 - 7\lambda + 12 - 2 = \lambda^2 - 7\lambda + 10 = 0$$
   $$(\lambda - 5)(\lambda - 2) = 0 \implies \lambda_1 = 5, \quad \lambda_2 = 2$$

   *Check Trace and Determinant:*
   - $\text{tr}(A) = 4 + 3 = 7 = 5 + 2$ $\checkmark$
   - $\det(A) = (4)(3) - (1)(2) = 10 = (5)(2)$ $\checkmark$

2. **Eigenvector for $\lambda_1 = 5$:**
   $$(A - 5I)v = 0 \implies \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies -v_1 + v_2 = 0 \implies v_1 = v_2$$
   $$v_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

3. **Eigenvector for $\lambda_2 = 2$:**
   $$(A - 2I)v = 0 \implies \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies 2v_1 + v_2 = 0 \implies v_2 = -2v_1$$
   $$v_2 = \begin{pmatrix} 1 \\ -2 \end{pmatrix}$$

---

## 2. Eigenspaces, Algebraic Multiplicity & Geometric Multiplicity

### 2.1 Eigenspaces and Multiplicities

For a given eigenvalue $\lambda$, the **eigenspace** $E_\lambda$ is the set of all eigenvectors corresponding to $\lambda$, together with the zero vector:
$$E_\lambda = N(A - \lambda I) = \{ v \in \mathbb{R}^n \mid (A - \lambda I) v = 0 \}$$
Because it is the null space of a matrix, $E_\lambda$ is a true subspace of $\mathbb{R}^n$.

- **Algebraic Multiplicity ($\text{AM}(\lambda)$):** The multiplicity of $\lambda$ as a root of the characteristic polynomial $p(\lambda) = 0$.
- **Geometric Multiplicity ($\text{GM}(\lambda)$):** The dimension of the eigenspace $E_\lambda$:
  $$\text{GM}(\lambda) = \dim(E_\lambda) = \text{nullity}(A - \lambda I) = n - \text{rank}(A - \lambda I)$$

#### Fundamental Multiplicity Inequality
**Theorem:** For every eigenvalue $\lambda$:
$$1 \le \text{GM}(\lambda) \le \text{AM}(\lambda)$$

#### Defective Matrices
A matrix is **defective** if there exists at least one eigenvalue where $\text{GM}(\lambda) < \text{AM}(\lambda)$. A defective matrix lacks a complete basis of $n$ independent eigenvectors and **cannot be diagonalized**.

```
Example of Defective vs Non-Defective Matrix (Both have AM(2) = 2):
Non-Defective: A = [ 2  0 ]                Defective: B = [ 2  1 ]
                   [ 0  2 ]                               [ 0  2 ]
Char Poly: (2 - lambda)^2 = 0              Char Poly: (2 - lambda)^2 = 0
A - 2I = [ 0  0 ]                         B - 2I = [ 0  1 ]
         [ 0  0 ]                                  [ 0  0 ]
Rank = 0 -> Nullity = GM = 2               Rank = 1 -> Nullity = GM = 1
GM(2) = AM(2) = 2 (Diagonalizable)         GM(2) = 1 < AM(2) = 2 (Defective!)
```

---

## 3. Matrix Diagonalization

### 3.1 Theory: Eigendecomposition $A = P D P^{-1}$

An $n\times n$ matrix $A$ is **diagonalizable** if it is similar to a diagonal matrix $D$:
$$A = P D P^{-1} \iff A P = P D$$
where $P = [v_1 \; v_2 \; \cdots \; v_n]$ contains $n$ linearly independent eigenvectors as columns, and $D = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$.

#### Diagonalizability Theorem
An $n\times n$ matrix $A$ is diagonalizable if and only if the sum of geometric multiplicities equals $n$:
$$\sum_{i} \text{GM}(\lambda_i) = n \iff \text{GM}(\lambda) = \text{AM}(\lambda) \quad \forall \lambda$$
*Corollary:* If $A$ has $n$ distinct eigenvalues, $A$ is automatically diagonalizable.

#### Efficient Matrix Powers
Diagonalization simplifies matrix exponentiation from repeated multiplications to scalar powers:
$$A^k = (P D P^{-1})^k = P D^k P^{-1} = P \begin{pmatrix} \lambda_1^k & 0 & \cdots & 0 \\ 0 & \lambda_2^k & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n^k \end{pmatrix} P^{-1}$$

---

### 3.2 Worked Example: Fast Matrix Power via Diagonalization

Using $A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$ from Section 1 with $\lambda_1 = 5, v_1 = (1, 1)^T$ and $\lambda_2 = 2, v_2 = (1, -2)^T$:

1. **Construct $P$ and $D$:**
   $$P = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix}, \qquad D = \begin{pmatrix} 5 & 0 \\ 0 & 2 \end{pmatrix}$$

2. **Compute $P^{-1}$:**
   $$\det(P) = (1)(-2) - (1)(1) = -3$$
   $$P^{-1} = -\frac{1}{3} \begin{pmatrix} -2 & -1 \\ -1 & 1 \end{pmatrix} = \frac{1}{3} \begin{pmatrix} 2 & 1 \\ 1 & -1 \end{pmatrix}$$

3. **Compute $A^k$:**
   $$A^k = P D^k P^{-1} = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix} \begin{pmatrix} 5^k & 0 \\ 0 & 2^k \end{pmatrix} \left( \frac{1}{3} \begin{pmatrix} 2 & 1 \\ 1 & -1 \end{pmatrix} \right)$$
   $$A^k = \frac{1}{3} \begin{pmatrix} 5^k & 2^k \\ 5^k & -2 \cdot 2^k \end{pmatrix} \begin{pmatrix} 2 & 1 \\ 1 & -1 \end{pmatrix} = \frac{1}{3} \begin{pmatrix} 2 \cdot 5^k + 2^k & 5^k - 2^k \\ 2 \cdot 5^k - 2 \cdot 2^k & 5^k + 2 \cdot 2^k \end{pmatrix}$$

   For $k=1$: $A^1 = \frac{1}{3}\begin{pmatrix} 10+2 & 5-2 \\ 10-4 & 5+4 \end{pmatrix} = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$ $\checkmark$.

---

## 4. Orthogonal Diagonalization & The Spectral Theorem

### 4.1 Real Symmetric Matrices and the Spectral Theorem

A real symmetric matrix satisfies $A = A^T$. Symmetric matrices play an indispensable role in physics, engineering, statistics, and machine learning (covariance and Hessian matrices).

#### The Spectral Theorem for Real Symmetric Matrices
Every real symmetric matrix $A \in \mathbb{R}^{n\times n}$ has the following properties:
1. All $n$ eigenvalues $\lambda_1, \dots, \lambda_n$ are **purely real numbers** ($\lambda_i \in \mathbb{R}$).
2. Eigenvectors corresponding to distinct eigenvalues are **strictly orthogonal**.
3. $A$ is **orthogonally diagonalizable**:
   $$A = Q \Lambda Q^T = \sum_{i=1}^n \lambda_i q_i q_i^T$$
   where $Q = [q_1 \; \dots \; q_n]$ is an orthogonal matrix ($Q^T Q = I$) containing orthonormal eigenvectors, and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$.

#### Proof: Eigenvalues of Real Symmetric Matrices are Real
Let $A \in \mathbb{R}^{n\times n}$ with $A = A^T$. Let $Av = \lambda v$ where $v \in \mathbb{C}^n, v \neq 0$.
Take the conjugate transpose (Hermitian conjugate) $v^H = (\bar{v})^T$:
$$v^H A v = v^H (\lambda v) = \lambda (v^H v) = \lambda \|v\|_2^2$$
Now conjugate transpose both sides of $A v = \lambda v$:
$$(A v)^H = (\lambda v)^H \implies v^H A^T = \bar{\lambda} v^H \implies v^H A = \bar{\lambda} v^H \quad (\text{since } A \text{ is real and symmetric})$$
Post-multiplying by $v$:
$$v^H A v = \bar{\lambda} (v^H v) = \bar{\lambda} \|v\|_2^2$$
Equating the two expressions:
$$\lambda \|v\|_2^2 = \bar{\lambda} \|v\|_2^2 \implies (\lambda - \bar{\lambda}) \|v\|_2^2 = 0$$
Since $v \neq 0$, $\|v\|_2^2 > 0$. Thus $\lambda = \bar{\lambda}$, proving $\lambda \in \mathbb{R}$. $\blacksquare$

#### Quadratic Forms and Matrix Definiteness
For a symmetric matrix $A$, the function $q(x) = x^T A x$ is a **quadratic form**.
- **Positive Definite ($A \succ 0$):** $x^T A x > 0$ for all $x \neq 0 \iff$ all $\lambda_i > 0$.
- **Positive Semi-Definite ($A \succeq 0$):** $x^T A x \ge 0$ for all $x \iff$ all $\lambda_i \ge 0$.
- **Negative Definite ($A \prec 0$):** $x^T A x < 0$ for all $x \neq 0 \iff$ all $\lambda_i < 0$.
- **Indefinite:** Has both positive and negative eigenvalues.

---

## 5. Singular Value Decomposition (SVD)

### 5.1 Universal Matrix Factorization $A = U \Sigma V^T$

While diagonalization $A = P D P^{-1}$ only applies to square, non-defective matrices, the **Singular Value Decomposition (SVD)** exists for **every** matrix $A \in \mathbb{R}^{m\times n}$ of any shape and rank:
$$A = U \Sigma V^T$$

```
Singular Value Decomposition Architecture:
    A (m x n)     =       U (m x m)      *      Sigma (m x n)     *      V^T (n x n)
  [           ]       [ |   |       | ]       [ sigma_1           ]       [ --- v_1^T --- ]
  [  Matrix   ]   =   [ u_1 u_2 ... u_m]  *   [        sigma_2    ]   *   [ --- v_2^T --- ]
  [           ]       [ |   |       | ]       [                   ]       [ --- v_n^T --- ]
                      (Left Singular Vecs)    (Singular Values >= 0)  (Right Singular Vecs)
```

- **$U \in \mathbb{R}^{m\times m}$:** Orthogonal matrix ($U^T U = I_m$). Columns $u_i$ are **left singular vectors** (eigenvectors of $A A^T$).
- **$V \in \mathbb{R}^{n\times n}$:** Orthogonal matrix ($V^T V = I_n$). Columns $v_i$ are **right singular vectors** (eigenvectors of $A^T A$).
- **$\Sigma \in \mathbb{R}^{m\times n}$:** Diagonal matrix containing non-negative **singular values** sorted in descending order:
  $$\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > \sigma_{r+1} = \cdots = 0, \quad \sigma_i = \sqrt{\lambda_i(A^T A)}$$

#### Geometric Meaning: Circle / Sphere to Hyper-Ellipsoid
Any linear transformation $x \mapsto Ax$ can be decomposed geometrically into three consecutive operations:
1. **Rotate / Reflect in Domain $\mathbb{R}^n$ ($V^T$):** Aligns input coordinate axes with principal axes.
2. **Scale along Coordinate Axes ($\Sigma$):** Stretches the $i$-th axis by factor $\sigma_i$. Transforms the unit sphere into a hyper-ellipsoid with semi-axis lengths $\sigma_1, \sigma_2, \dots, \sigma_r$.
3. **Rotate / Reflect in Codomain $\mathbb{R}^m$ ($U$):** Orients the hyper-ellipsoid into the output space.

```
Geometric Transformation Pipeline of SVD:
   Unit Circle in R^2           Stretched by Sigma           Rotated in R^2 (or R^m)
         y ^                          y ^                          y ^
           |                            |                            |       /---/ (sigma_1 u_1)
       1 +---+                      1 +---+                          |      /   /
         |   |      -- V^T -->        |   |        -- U -->          |     /   /
      ---+---+---> x               ---+---+---> x                 ---+---+----> x
        -1   1                       -3   3                         /   /
                                 (sigma_1=3, sigma_2=1)            /---/ (sigma_2 u_2)
```

#### Compact / Economy SVD and Outer Product Expansion
If $\text{rank}(A) = r \le \min(m, n)$:
$$A = U_r \Sigma_r V_r^T = \sum_{i=1}^r \sigma_i u_i v_i^T$$
where $u_i v_i^T$ is a rank-1 matrix with $\|u_i v_i^T\|_2 = 1$.

#### Low-Rank Matrix Approximation (Eckart–Young–Mirsky Theorem)
The optimal rank-$k$ approximation ($k < r$) of $A$ minimizing reconstruction error is obtained by truncating the SVD:
$$A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$$
$$\min_{\text{rank}(B)=k} \|A - B\|_F^2 = \|A - A_k\|_F^2 = \sum_{i=k+1}^r \sigma_i^2$$

#### The Moore–Penrose Pseudoinverse $A^+$
For any matrix $A = U \Sigma V^T$, the pseudoinverse is:
$$A^+ = V \Sigma^+ U^T$$
where $\Sigma^+$ replaces every non-zero singular value $\sigma_i$ with $1/\sigma_i$ and transposes. The minimum-norm least-squares solution is given by:
$$\hat{x} = A^+ b$$

---

## 6. Principal Component Analysis (PCA)

### 6.1 Dimensionality Reduction and Variance Maximization

Principal Component Analysis (PCA) is an unsupervised learning technique that finds orthogonal axes of **maximum variance** in high-dimensional data.

```
PCA Geometric Concept:
      y ^
        |            *   *     / PC 1 (Max Variance, lambda_1)
        |         *   *   *  /
        |       *   *   *  /
        |     *   *   *  /
        |   *   *   *  /   \
        | *   *   *  /       \  PC 2 (Orthogonal, lambda_2)
        +-------------------------> x
```

#### The PCA Algorithm Step-by-Step
1. **Data Matrix:** Let $X \in \mathbb{R}^{N \times d}$ contain $N$ observations of $d$ features.
2. **Mean Centering:** Compute sample mean $\mu = \frac{1}{N} \sum_{i=1}^N x_i$. Subtract mean from each row:
   $$X_c = X - \mathbf{1} \mu^T$$
3. **Sample Covariance Matrix:**
   $$C = \frac{1}{N-1} X_c^T X_c \in \mathbb{R}^{d\times d}$$
4. **Spectral Decomposition:** Orthogonally diagonalize $C$:
   $$C = Q \Lambda Q^T, \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_d), \quad \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$$
   The eigenvectors $q_1, \dots, q_d$ are the **principal component loading vectors**.
5. **Project Data:** Project centered data onto top $k$ components ($k < d$):
   $$Z = X_c Q_k \in \mathbb{R}^{N \times k}, \quad \text{where } Q_k = [q_1 \; q_2 \; \cdots \; q_k]$$

#### Proportion of Variance Explained & Scree Plot
The variance captured by the $i$-th principal component is $\lambda_i$.
$$\text{Proportion of Variance Explained (PVE)}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$
$$\text{Cumulative PVE}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

#### Fast PCA via SVD of Centered Data Matrix
Instead of forming $C = \frac{1}{N-1} X_c^T X_c$ (which squares the condition number), compute the economy SVD:
$$\frac{1}{\sqrt{N-1}} X_c = U \Sigma V^T$$
Then $V = Q$ (principal components) and $\lambda_i = \sigma_i^2$. This is computationally faster and numerically superior.

---

## 7. Iterative Methods for Linear Systems (Jacobi, Gauss–Seidel & SOR)

### 7.1 Theory: Stationary Iterative Splittings

For massive sparse systems $Ax = b$ (e.g., $10^6$ equations from PDE discretization), direct elimination ($O(n^3)$) is impossible. **Iterative methods** compute a sequence $x^{(0)}, x^{(1)}, x^{(2)}, \dots \to x^*$.

Split matrix $A = D + L + U$:
- $D$: Diagonal part of $A$.
- $L$: Strictly lower triangular part of $A$.
- $U$: Strictly upper triangular part of $A$.

```
Matrix Splitting A = D + L + U:
   [ a_11  a_12  a_13 ]       [ a_11   0     0   ]     [  0    0    0  ]     [  0   a_12 a_13 ]
   [ a_21  a_22  a_23 ]   =   [  0   a_22    0   ]  +  [ a_21  0    0  ]  +  [  0    0   a_23 ]
   [ a_31  a_32  a_33 ]       [  0     0   a_33  ]     [ a_31 a_32  0  ]     [  0    0    0   ]
           A                         D                     L                     U
```

#### 1. Jacobi Method
Update each coordinate using only values from the **previous iteration**:
$$D x^{(k+1)} = b - (L + U) x^{(k)} \implies x^{(k+1)} = D^{-1} b - D^{-1}(L + U) x^{(k)}$$
Componentwise formula:
$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x_j^{(k)} \right)$$

#### 2. Gauss–Seidel Method
Immediately uses **newly updated** components $x_1^{(k+1)}, \dots, x_{i-1}^{(k+1)}$ within the same iteration:
$$(D + L) x^{(k+1)} = b - U x^{(k)} \implies x^{(k+1)} = (D + L)^{-1} b - (D + L)^{-1} U x^{(k)}$$
Componentwise formula:
$$x_i^{(k+1)} = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

#### 3. Successive Over-Relaxation (SOR)
Accelerates Gauss–Seidel using a relaxation parameter $\omega \in (0, 2)$:
$$x_i^{(k+1)} = (1 - \omega) x_i^{(k)} + \frac{\omega}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x_j^{(k+1)} - \sum_{j > i} a_{ij} x_j^{(k)} \right)$$

#### Convergence Criterion: Spectral Radius $\rho(T) < 1$
Every stationary iterative method can be written as $x^{(k+1)} = T x^{(k)} + c$. The error vector $e^{(k)} = x^{(k)} - x^*$ satisfies $e^{(k)} = T^k e^{(0)}$.
**Theorem:** The iteration converges for any initial guess $x^{(0)}$ if and only if the **spectral radius** satisfies:
$$\rho(T) = \max_i |\lambda_i(T)| < 1$$

#### Sufficient Condition: Strict Diagonal Dominance
If $A$ is **strictly diagonally dominant**:
$$|a_{ii}| > \sum_{j \neq i} |a_{ij}| \quad \forall i = 1, \dots, n$$
then both Jacobi and Gauss–Seidel methods are **guaranteed to converge**.

---

### 7.2 Worked Example: Jacobi vs Gauss–Seidel Iteration

Solve the system:
$$\begin{aligned}
10 x_1 + x_2 &= 11 \\
x_1 + 10 x_2 &= 11
\end{aligned}$$
Exact solution is $x_1 = 1, x_2 = 1$. Start with initial guess $x^{(0)} = (0, 0)^T$.

1. **Jacobi Iteration:**
   $$x_1^{(k+1)} = \frac{11 - x_2^{(k)}}{10}, \qquad x_2^{(k+1)} = \frac{11 - x_1^{(k)}}{10}$$
   - $k=0 \to 1$: $x_1^{(1)} = \frac{11-0}{10} = 1.1$, $x_2^{(1)} = \frac{11-0}{10} = 1.1$.
   - $k=1 \to 2$: $x_1^{(2)} = \frac{11-1.1}{10} = 0.99$, $x_2^{(2)} = \frac{11-1.1}{10} = 0.99$.
   - $k=2 \to 3$: $x_1^{(3)} = \frac{11-0.99}{10} = 1.001$, $x_2^{(3)} = \frac{11-0.99}{10} = 1.001$.

2. **Gauss–Seidel Iteration:**
   $$x_1^{(k+1)} = \frac{11 - x_2^{(k)}}{10}, \qquad x_2^{(k+1)} = \frac{11 - x_1^{(k+1)}}{10}$$
   - $k=0 \to 1$: $x_1^{(1)} = \frac{11-0}{10} = 1.1$, $x_2^{(1)} = \frac{11-1.1}{10} = 0.99$.
   - $k=1 \to 2$: $x_1^{(2)} = \frac{11-0.99}{10} = 1.001$, $x_2^{(2)} = \frac{11-1.001}{10} = 0.9999$.
   Gauss–Seidel reaches 4 decimal places in half the iterations of Jacobi!

---

## 8. The Power Method & Eigenvalue Algorithms

### 8.1 Algorithms: Power, Inverse, and Shifted Power Methods

#### 1. The Standard Power Method (Dominant Eigenvalue)
Finds the eigenvalue of largest absolute magnitude $|\lambda_1| > |\lambda_2| \ge \dots \ge |\lambda_n|$:
1. Choose random non-zero vector $x^{(0)}$.
2. For $k = 0, 1, 2, \dots$:
   $$y^{(k+1)} = A x^{(k)}$$
   $$x^{(k+1)} = \frac{y^{(k+1)}}{\|y^{(k+1)}\|_2}$$
3. Estimate eigenvalue via the **Rayleigh Quotient**:
   $$\lambda^{(k+1)} = \frac{(x^{(k+1)})^T A x^{(k+1)}}{(x^{(k+1)})^T x^{(k+1)}} = (x^{(k+1)})^T A x^{(k+1)}$$

- **Convergence Rate:** The vector error converges at linear rate $\mathcal{O}\left( \left|\frac{\lambda_2}{\lambda_1}\right|^k \right)$. The Rayleigh quotient converges quadratically at rate $\mathcal{O}\left( \left|\frac{\lambda_2}{\lambda_1}\right|^{2k} \right)$.

#### 2. The Inverse Power Method (Smallest Eigenvalue)
To find the smallest eigenvalue $\lambda_n$ of an invertible matrix $A$:
Apply the power method to $A^{-1}$, since eigenvalues of $A^{-1}$ are $1/\lambda_i$.
Solve $A y^{(k+1)} = x^{(k)}$ at each step via pre-factored LU decomposition.

#### 3. The Shifted Inverse Power Method (Targeted Eigenvalue)
To find the eigenvalue closest to a target scalar $\mu \in \mathbb{R}$:
Apply the power method to $(A - \mu I)^{-1}$. The dominant eigenvalue of $(A - \mu I)^{-1}$ is $\frac{1}{\lambda_j - \mu}$, which is largest when $\lambda_j \approx \mu$.
This converges extremely rapidly when $\mu$ is close to an eigenvalue.

---

## 9. Linear Transformations: Rotation and General Transforms

### 9.1 Matrix Representations of Geometric Mappings

A mapping $T: \mathbb{R}^n \to \mathbb{R}^m$ is a **linear transformation** if:
1. $T(u + v) = T(u) + T(v)$
2. $T(c u) = c T(u)$ for all $c \in \mathbb{R}$

The standard matrix representation of $T$ is constructed by evaluating $T$ on the standard basis vectors:
$$[T] = [T(e_1) \; T(e_2) \; \cdots \; T(e_n)] \in \mathbb{R}^{m\times n}$$

#### 2D Rotation Matrix ($R_\theta$)
Rotating vectors counterclockwise by angle $\theta$:
$$T(e_1) = \begin{pmatrix} \cos\theta \\ \sin\theta \end{pmatrix}, \quad T(e_2) = \begin{pmatrix} -\sin\theta \\ \cos\theta \end{pmatrix} \implies R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$
- Determinant: $\det(R_\theta) = \cos^2\theta + \sin^2\theta = 1$.
- Orthogonality: $R_\theta^T R_\theta = I \implies R_\theta^{-1} = R_\theta^T = R_{-\theta}$.
- Eigenvalues: $\lambda = \cos\theta \pm i \sin\theta = e^{\pm i \theta}$.

#### 3D Elementary Rotation Matrices
- Rotation about $x$-axis by $\alpha$:
  $$R_x(\alpha) = \begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos\alpha & -\sin\alpha \\ 0 & \sin\alpha & \cos\alpha \end{pmatrix}$$
- Rotation about $y$-axis by $\beta$:
  $$R_y(\beta) = \begin{pmatrix} \cos\beta & 0 & \sin\beta \\ 0 & 1 & 0 \\ -\sin\beta & 0 & \cos\beta \end{pmatrix}$$
- Rotation about $z$-axis by $\gamma$:
  $$R_z(\gamma) = \begin{pmatrix} \cos\gamma & -\sin\gamma & 0 \\ \sin\gamma & \cos\gamma & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

---

## 10. The Leslie Demography Model (Population Dynamics)

### 10.1 Mathematical Formulation of Age-Structured Models

The **Leslie matrix** models the growth and age-distribution dynamics of a female population categorized into $n$ discrete age classes.

Let $x^{(k)} = (x_1^{(k)}, x_2^{(k)}, \dots, x_n^{(k)})^T$ be the population vector at time step $k$.
- $f_i \ge 0$: Age-specific **fecundity/fertility rate** (average number of offspring per female in age class $i$).
- $s_i \in (0, 1]$: Age-specific **survival probability** from age class $i$ to $i+1$.

```
Leslie Matrix Structure:
       [  f_1    f_2    f_3   ...   f_n-1    f_n  ]   <- Top Row: Fertility Rates
       [  s_1     0      0    ...     0       0   ]   <- Subdiagonal: Survival Rates
   L = [   0     s_2     0    ...     0       0   ]
       [   0      0     s_3   ...     0       0   ]
       [   0      0      0    ...    s_n-1    0   ]
```

Population transition equation:
$$x^{(k+1)} = L x^{(k)} \implies x^{(k)} = L^k x^{(0)}$$

#### Perron–Frobenius Theorem & Long-Term Asymptotic Behavior
Because $L \ge 0$ (non-negative matrix), the **Perron–Frobenius Theorem** guarantees:
1. There exists a unique real positive dominant eigenvalue $\lambda_1 > 0$.
2. The corresponding eigenvector $v_1$ has strictly positive components ($v_1 > 0$).
3. As $k \to \infty$, the population grows exponentially at rate $\lambda_1$:
   $$x^{(k)} \approx c_1 \lambda_1^k v_1$$
   - **$\lambda_1 > 1$:** Population grows indefinitely.
   - **$\lambda_1 = 1$:** Stable, stationary population size.
   - **$\lambda_1 < 1$:** Population declines toward extinction.
4. The normalized eigenvector $\frac{v_1}{\|v_1\|_1}$ represents the **stable age distribution** (fraction of population in each age class).

---

### 10.2 Worked Example: Leslie Population Projection

Consider a 3-stage population model with $L = \begin{pmatrix} 0 & 4 & 2 \\ 0.5 & 0 & 0 \\ 0 & 0.25 & 0 \end{pmatrix}$ and initial population $x^{(0)} = (100, 40, 20)^T$.

1. **Calculate Population at $k=1$ ($x^{(1)} = L x^{(0)}$):**
   $$x^{(1)} = \begin{pmatrix} 0 & 4 & 2 \\ 0.5 & 0 & 0 \\ 0 & 0.25 & 0 \end{pmatrix} \begin{pmatrix} 100 \\ 40 \\ 20 \end{pmatrix} = \begin{pmatrix} 0(100) + 4(40) + 2(20) \\ 0.5(100) + 0 + 0 \\ 0 + 0.25(40) + 0 \end{pmatrix} = \begin{pmatrix} 160 + 40 \\ 50 \\ 10 \end{pmatrix} = \begin{pmatrix} 200 \\ 50 \\ 10 \end{pmatrix}$$
   Total population increased from $160 \to 260$.

2. **Compute Dominant Growth Rate $\lambda_1$:**
   $$\det(L - \lambda I) = \begin{vmatrix} -\lambda & 4 & 2 \\ 0.5 & -\lambda & 0 \\ 0 & 0.25 & -\lambda \end{vmatrix} = -\lambda(\lambda^2) - 4(-0.5\lambda) + 2(0.5 \times 0.25) = -\lambda^3 + 2\lambda + 0.25 = 0$$
   $$\lambda^3 - 2\lambda - 0.25 = 0$$
   Testing $\lambda \approx 1.465$: $(1.465)^3 - 2(1.465) - 0.25 = 3.144 - 2.930 - 0.25 \approx 0$.
   Dominant growth rate is $\lambda_1 \approx 1.465$ (46.5% growth per cycle).

---

## 11. Linear Programming Problems (LPP) & Simplex Foundations

### 11.1 Theory: Optimization on Convex Polytopes

A Linear Programming Problem (LPP) seeks to maximize or minimize a linear objective function subject to linear equality and inequality constraints:
$$\begin{aligned}
\text{Maximize } & z = c^T x = \sum_{j=1}^n c_j x_j \\
\text{Subject to } & A x \le b \\
& x \ge 0
\end{aligned}$$

```
Feasible Polytope and Simplex Path to Optimal Vertex:
       x_2 ^
           |       Constraint 1
           |        \
           |   (0,6) +--------+ (2,6)  <--- Optimal Solution (Max z)
           |         |Feasible|  \
           |         | Region |   + (4,3)
           |         |        |    \ Constraint 2
           |   (0,0) +--------+-----+--------> x_1
                    (Origin) (4,0)
             Simplex Path: (0,0) -> (4,0) -> (4,3) -> (2,6)*
```

#### Fundamental Theorem of Linear Programming
1. The feasible region $S = \{x \in \mathbb{R}^n \mid Ax \le b, x \ge 0\}$ is a **convex polytope**.
2. If an optimal solution exists, at least one optimal solution occurs at an **extreme point (vertex/corner)** of the feasible region.

#### Standard Equality Form via Slack Variables
To apply algebraic solvers (Simplex algorithm), convert inequalities $Ax \le b$ into equalities by adding non-negative **slack variables** $s_i \ge 0$:
$$A x + I s = b, \quad x \ge 0, \; s \ge 0$$

#### The Simplex Tableau Algorithm Concept
1. Set up initial tableau with non-basic variables $x = 0$ (starting at origin vertex $s = b$).
2. **Optimality Test:** If all coefficients in the objective row are non-negative ($\ge 0$), the current basic feasible solution is optimal.
3. **Pivot Column (Entering Variable):** Choose column with the most negative objective coefficient (steepest rate of improvement).
4. **Pivot Row (Leaving Variable):** Perform the **Minimum Ratio Test** $\min_{i: a_{ik} > 0} \frac{b_i}{a_{ik}}$ to maintain feasibility.
5. **Pivot Operation:** Use elementary row operations to reduce the pivot column to a standard unit basis vector. Repeat.

---

### 11.2 Worked Example: 2-Variable Graphical and Corner-Point LPP

**Problem:**
$$\begin{aligned}
\text{Maximize } & z = 3x_1 + 5x_2 \\
\text{Subject to: } & x_1 \le 4 \\
& 2x_2 \le 12 \implies x_2 \le 6 \\
& 3x_1 + 2x_2 \le 18 \\
& x_1 \ge 0, \; x_2 \ge 0
\end{aligned}$$

1. **Find Vertices of Feasible Region:**
   - Origin: $(0, 0)$
   - $x_1$-intercept: $(4, 0)$
   - Intersection of $x_1 = 4$ and $3x_1 + 2x_2 = 18$:
     $$3(4) + 2x_2 = 18 \implies 2x_2 = 6 \implies x_2 = 3 \implies (4, 3)$$
   - Intersection of $x_2 = 6$ and $3x_1 + 2x_2 = 18$:
     $$3x_1 + 2(6) = 18 \implies 3x_1 = 6 \implies x_1 = 2 \implies (2, 6)$$
   - $x_2$-intercept: $(0, 6)$

2. **Evaluate Objective Function $z = 3x_1 + 5x_2$ at Every Vertex:**

| Vertex $(x_1, x_2)$ | Objective Calculation $z = 3x_1 + 5x_2$ | Value |
|---|---|---|
| $(0, 0)$ | $3(0) + 5(0)$ | $0$ |
| $(4, 0)$ | $3(4) + 5(0)$ | $12$ |
| $(4, 3)$ | $3(4) + 5(3) = 12 + 15$ | $27$ |
| **$(2, 6)$** | **$3(2) + 5(6) = 6 + 30$** | **$36$ (OPTIMAL)** |
| $(0, 6)$ | $3(0) + 5(6)$ | $30$ |

**Conclusion:** The maximum value is $z^* = 36$, achieved at $x_1^* = 2, x_2^* = 6$.

---

## 12. Comprehensive Solved Exam-Style Problems

### Problem 1: Complete Spectral Decomposition of a Symmetric Matrix
**Statement:** Orthogonally diagonalize $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$.

**Step-by-step Solution:**
1. **Eigenvalues:**
   $$\det(A - \lambda I) = (3 - \lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 4)(\lambda - 2) = 0 \implies \lambda_1 = 4, \; \lambda_2 = 2$$
2. **Eigenvector for $\lambda_1 = 4$:**
   $$(A - 4I)v = 0 \implies \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0 \implies v_1 = v_2 \implies v_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$
   Normalize: $q_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.
3. **Eigenvector for $\lambda_2 = 2$:**
   $$(A - 2I)v = 0 \implies \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix} = 0 \implies v_1 = -v_2 \implies v_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$
   Normalize: $q_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$.
4. **Orthogonal Factorization:**
   $$Q = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}, \quad \Lambda = \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix}, \quad A = Q \Lambda Q^T$$

---

### Problem 2: Full SVD Computation
**Statement:** Find the singular value decomposition $A = U \Sigma V^T$ of $A = \begin{pmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{pmatrix}$.

**Step-by-step Solution:**
1. **Compute $A A^T$ (size $2\times 2$):**
   $$A A^T = \begin{pmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{pmatrix} \begin{pmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{pmatrix} = \begin{pmatrix} 9+4+4 & 6+6-4 \\ 6+6-4 & 4+9+4 \end{pmatrix} = \begin{pmatrix} 17 & 8 \\ 8 & 17 \end{pmatrix}$$
2. **Eigenvalues of $A A^T$:**
   $$\det(A A^T - \lambda I) = (17 - \lambda)^2 - 64 = 0 \implies 17 - \lambda = \pm 8 \implies \lambda_1 = 25, \; \lambda_2 = 9$$
   Singular values: $\sigma_1 = \sqrt{25} = 5, \; \sigma_2 = \sqrt{9} = 3$.
   $$\Sigma = \begin{pmatrix} 5 & 0 & 0 \\ 0 & 3 & 0 \end{pmatrix}$$
3. **Left Singular Vectors $U$ (from $A A^T$):**
   - For $\lambda_1 = 25$: $\begin{pmatrix} -8 & 8 \\ 8 & -8 \end{pmatrix} u_1 = 0 \implies u_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$.
   - For $\lambda_2 = 9$: $\begin{pmatrix} 8 & 8 \\ 8 & 8 \end{pmatrix} u_2 = 0 \implies u_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$.
   $$U = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$
4. **Right Singular Vectors $v_i = \frac{1}{\sigma_i} A^T u_i$:**
   $$v_1 = \frac{1}{5} \begin{pmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{pmatrix} = \frac{1}{5\sqrt{2}} \begin{pmatrix} 5 \\ 5 \\ 0 \end{pmatrix} = \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{pmatrix}$$
   $$v_2 = \frac{1}{3} \begin{pmatrix} 3 & 2 \\ 2 & 3 \\ 2 & -2 \end{pmatrix} \begin{pmatrix} 1/\sqrt{2} \\ -1/\sqrt{2} \end{pmatrix} = \frac{1}{3\sqrt{2}} \begin{pmatrix} 1 \\ -1 \\ 4 \end{pmatrix} = \begin{pmatrix} 1/(3\sqrt{2}) \\ -1/(3\sqrt{2}) \\ 4/(3\sqrt{2}) \end{pmatrix}$$
   $v_3$ is the cross product $v_1 \times v_2 = \begin{pmatrix} 2/3 \\ -2/3 \\ -1/3 \end{pmatrix}$.

---

## 13. Unit II Summary & Formula Cheat Sheet

| Mathematical Concept | Defining Formula / Key Theorem | Computational Role |
|---|---|---|
| **Eigenvalues / Eigenvectors** | $Av = \lambda v, \; \det(A - \lambda I) = 0$ | Natural scaling axes; simplifies dynamical systems |
| **Diagonalization** | $A = P D P^{-1} \implies A^k = P D^k P^{-1}$ | Computes matrix powers in $O(n)$ flops |
| **Spectral Theorem** | $A = Q \Lambda Q^T$ ($A = A^T$, real $\lambda_i$) | Orthogonal axes for symmetric/covariance matrices |
| **Singular Value Decomposition** | $A = U \Sigma V^T$ | Universal factorization; pseudoinverse and low-rank compression |
| **PCA** | $C = \frac{1}{N-1}X_c^T X_c = Q \Lambda Q^T$ | Finds axes of maximum variance for dimensionality reduction |
| **Jacobi Iteration** | $x^{(k+1)} = D^{-1}(b - (L+U)x^{(k)})$ | Parallelizable sparse solver; requires diagonal dominance |
| **Gauss–Seidel Iteration** | $x^{(k+1)} = (D+L)^{-1}(b - Ux^{(k)})$ | In-place iterative solver; faster convergence |
| **Power Method** | $x^{(k+1)} = \frac{A x^{(k)}}{\|A x^{(k)}\|_2}$ | Finds dominant eigenvalue $\lambda_1$ and eigenvector |
| **Leslie Demography Model** | $x^{(k+1)} = L x^{(k)}, \; \lambda_1 > 1 \implies \text{growth}$ | Age-structured population dynamics and stable age profile |
| **Linear Programming (LPP)** | $\max c^T x \text{ s.t. } Ax \le b, x \ge 0$ | Optimization on convex polytopes; optimum at vertex |
