# Unit I — Vector Spaces, Orthogonality & Decompositions
### 25MAT532A — Computational Linear Algebra

This unit builds the entire computational and theoretical toolkit of linear algebra from first principles: matrices and their structural properties $\to$ determinants and inverses $\to$ solving linear systems via Gauss elimination $\to$ vector spaces, subspaces, and the four fundamental subspaces $\to$ linear independence, bases, and coordinate changes $\to$ inner products, vector metrics, and orthogonality $\to$ Gram–Schmidt orthogonalization $\to$ orthogonal projections $\to$ the least-squares principle $\to$ matrix factorizations (LU, Cholesky, and QR) that make large-scale computation fast and numerically stable.

---

## Table of Contents
1. [Matrices and Their Properties](#1-matrices-and-their-properties)
2. [Determinants and Inverses](#2-determinants-and-inverses)
3. [Vector Metrics and Geometry](#3-vector-metrics-and-geometry)
4. [Gauss Elimination and Systems of Linear Equations](#4-gauss-elimination-and-systems-of-linear-equations)
5. [Vector Spaces and Subspaces](#5-vector-spaces-and-subspaces)
6. [The Four Fundamental Subspaces](#6-the-four-fundamental-subspaces)
7. [Linear Independence, Basis, and Dimension](#7-linear-independence-basis-and-dimension)
8. [Elementary Matrices and Inversion](#8-elementary-matrices-and-inversion)
9. [LU and Cholesky Decompositions](#9-lu-and-cholesky-decompositions)
10. [Change of Basis and Coordinate Transformations](#10-change-of-basis-and-coordinate-transformations)
11. [Inner Product Spaces and Orthogonality](#11-inner-product-spaces-and-orthogonality)
12. [The Gram–Schmidt Orthogonalization Process](#12-the-gramschmidt-orthogonalization-process)
13. [Orthogonal Projection onto a Subspace](#13-orthogonal-projection-onto-a-subspace)
14. [The Least-Squares Principle and Regression](#14-the-least-squares-principle-and-regression)
15. [QR Decomposition](#15-qr-decomposition)
16. [Comprehensive Solved Exam-Style Problems](#16-comprehensive-solved-exam-style-problems)
17. [Unit I Summary & Formula Cheat Sheet](#17-unit-i-summary--formula-cheat-sheet)

---

## 1. Matrices and Their Properties

### 1.1 Mathematical Theory & Geometric Intuition

A matrix is a rectangular array of real numbers, $A \in \mathbb{R}^{m\times n}$, having $m$ rows and $n$ columns:
$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

To master linear algebra, you must seamlessly navigate **three complementary perspectives**:

```
+-------------------------------------------------------------------------+
|                         THREE VIEWS OF A MATRIX                         |
+-------------------------------------------------------------------------+
| 1. Table of Numbers     | Data storage, entrywise indexing A_ij         |
| 2. Collection of Columns| A = [a_1  a_2 ... a_n], vectors in R^m        |
| 3. Linear Transformation| Mapping T_A: R^n -> R^m, where x |-> Ax       |
+-------------------------------------------------------------------------+
```

#### Fundamental Matrix Operations

1. **Matrix Addition & Scalar Multiplication:**
   Given $A, B \in \mathbb{R}^{m\times n}$ and $c \in \mathbb{R}$:
   $$(A + B)_{ij} = A_{ij} + B_{ij}, \qquad (cA)_{ij} = c A_{ij}$$

2. **Matrix Multiplication (Composition of Transformations):**
   For $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$, the product $C = AB \in \mathbb{R}^{m\times n}$ is defined by:
   $$C_{ij} = (AB)_{ij} = \sum_{p=1}^k A_{ip} B_{pj}$$
   
   - **Column-combination interpretation:** The $j$-th column of $AB$ is a linear combination of the columns of $A$ with weights given by the $j$-th column of $B$:
     $$\text{col}_j(AB) = A \cdot \text{col}_j(B) = \sum_{p=1}^k B_{pj} a_p$$
   - **Row-combination interpretation:** The $i$-th row of $AB$ is a linear combination of the rows of $B$ with weights from the $i$-th row of $A$:
     $$\text{row}_i(AB) = \text{row}_i(A) \cdot B = \sum_{p=1}^k A_{ip} \text{row}_p(B)$$

3. **Matrix Transpose:**
   $(A^T)_{ij} = A_{ji}$. The transpose flips rows and columns across the principal diagonal.
   - Property: $(AB)^T = B^T A^T$ (the order of operations reverses).
   - Property: $(A + B)^T = A^T + B^T$.
   - Property: $(A^T)^T = A$.

4. **Matrix Trace:**
   For a square matrix $A \in \mathbb{R}^{n\times n}$, the trace is the sum of diagonal entries:
   $$\text{tr}(A) = \sum_{i=1}^n A_{ii}$$
   - Cyclic property: $\text{tr}(AB) = \text{tr}(BA)$ for any $A \in \mathbb{R}^{m\times n}, B \in \mathbb{R}^{n\times m}$.
   - Linearity: $\text{tr}(\alpha A + \beta B) = \alpha\,\text{tr}(A) + \beta\,\text{tr}(B)$.

#### Special Matrix Families
- **Identity Matrix ($I_n$):** Neutral element for multiplication, $I_n x = x$.
- **Diagonal Matrix ($D$):** $D_{ij} = 0$ for all $i \neq j$. Scales axes independently.
- **Symmetric Matrix:** $A = A^T$. Exhibits orthogonal eigenvectors and real eigenvalues (crucial for Unit II).
- **Skew-Symmetric Matrix:** $A = -A^T$. Diagonal entries are identically zero ($A_{ii} = -A_{ii} \implies A_{ii}=0$).
- **Orthogonal Matrix ($Q$):** $Q^T Q = Q Q^T = I$. Columns and rows form orthonormal sets; preserves Euclidean norms and angles ($\|Qx\|_2 = \|x\|_2$).

#### Rigorous Proof: Transpose of a Product $(AB)^T = B^T A^T$
Let $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$. Consider the $(i, j)$-th entry of $(AB)^T$:
$$[(AB)^T]_{ij} = (AB)_{ji} = \sum_{p=1}^k A_{jp} B_{pi}$$
Now consider the $(i, j)$-th entry of $B^T A^T$:
$$[B^T A^T]_{ij} = \sum_{p=1}^k (B^T)_{ip} (A^T)_{pj} = \sum_{p=1}^k B_{pi} A_{jp} = \sum_{p=1}^k A_{jp} B_{pi}$$
Since scalar multiplication in $\mathbb{R}$ commutes, both expressions are identical for all $1 \le i \le n$ and $1 \le j \le m$. Thus, $(AB)^T = B^T A^T$. $\blacksquare$

---

### 1.2 Step-by-Step Worked Example

Let $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$ and $B = \begin{pmatrix} 2 & 0 \\ -1 & 5 \end{pmatrix}$.

1. **Addition:**
   $$A + B = \begin{pmatrix} 1+2 & 2+0 \\ 3+(-1) & 4+5 \end{pmatrix} = \begin{pmatrix} 3 & 2 \\ 2 & 9 \end{pmatrix}$$

2. **Multiplication $AB$:**
   $$AB = \begin{pmatrix} (1)(2) + (2)(-1) & (1)(0) + (2)(5) \\ (3)(2) + (4)(-1) & (3)(0) + (4)(5) \end{pmatrix} = \begin{pmatrix} 2 - 2 & 0 + 10 \\ 6 - 4 & 0 + 20 \end{pmatrix} = \begin{pmatrix} 0 & 10 \\ 2 & 20 \end{pmatrix}$$

3. **Multiplication $BA$ (demonstrating non-commutativity):**
   $$BA = \begin{pmatrix} (2)(1) + (0)(3) & (2)(2) + (0)(4) \\ (-1)(1) + (5)(3) & (-1)(2) + (5)(4) \end{pmatrix} = \begin{pmatrix} 2 & 4 \\ 14 & 18 \end{pmatrix} \neq AB$$

4. **Transpose Product Verification:**
   $$A^T = \begin{pmatrix} 1 & 3 \\ 2 & 4 \end{pmatrix}, \quad B^T = \begin{pmatrix} 2 & -1 \\ 0 & 5 \end{pmatrix}$$
   $$B^T A^T = \begin{pmatrix} (2)(1) + (-1)(2) & (2)(3) + (-1)(4) \\ (0)(1) + (5)(2) & (0)(3) + (5)(4) \end{pmatrix} = \begin{pmatrix} 0 & 2 \\ 10 & 20 \end{pmatrix} = (AB)^T$$

---

### 1.3 Nuances, Pitfalls & Computational Insights

- **Matrix Multiplication Associativity vs Commutativity:** While $AB \neq BA$ in general, multiplication is strictly associative: $A(BC) = (AB)C$.
- **Flop Complexity of Matrix Multiplication:** For $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$, computing $AB$ requires $2mkn$ floating-point operations ($mkn$ multiplications and $mkn$ additions). 
- **Matrix-Vector Evaluation Order:** If computing $A B x$ where $A, B \in \mathbb{R}^{n\times n}$ and $x \in \mathbb{R}^n$, computing $A(Bx)$ takes $O(n^2)$ flops, whereas $(AB)x$ takes $O(n^3)$ flops. **Always multiply from right to left when vectors are involved!**

---

### 1.4 Real-World Application: Artificial Intelligence & Graphics
In deep neural networks, a forward propagation step across a layer is expressed as $y = \sigma(W x + b)$, where $W$ is the weight matrix, $x$ is the input activation vector, $b$ is the bias vector, and $\sigma(\cdot)$ is an elementwise activation function. The composition of $L$ layers is $f(x) = \sigma(W_L \cdots \sigma(W_2 \sigma(W_1 x + b_1) + b_2)\cdots + b_L)$. In 3D rendering engines (e.g., OpenGL/DirectX), 3D affine transformations (rotation, scaling, translation) are expressed as $4\times 4$ homogeneous matrices composed via matrix multiplication.

---

## 2. Determinants and Inverses

### 2.1 Mathematical Theory

The determinant is a scalar-valued function $\det: \mathbb{R}^{n\times n} \to \mathbb{R}$ that measures the **signed scaling factor of oriented volume** under the transformation $x \mapsto Ax$.

```
Geometric Area Scaling in 2D:
   y ^                       y ^
     |                         |        /--------/ (a+c, b+d)
   1 +-----+                   |       /        /
     |     |   --- A --->      |      /        /  Area = |det(A)|
     |     |                   |     /        /       = |ad - bc|
   0 +-----+--> x              |    +--------+--> x
     0     1                       (0,0)   (a,b)
   Unit Square (Area = 1)          Parallelogram formed by cols of A
```

#### Core Properties of Determinants
1. $\det(I_n) = 1$.
2. $\det(AB) = \det(A)\det(B)$ (volume scaling composes multiplicatively).
3. $\det(A^T) = \det(A)$.
4. $\det(A^{-1}) = \frac{1}{\det(A)}$ (for invertible $A$).
5. $\det(cA) = c^n \det(A)$ for an $n\times n$ matrix $A$ and scalar $c$.
6. Swapping two rows/columns multiplies the determinant by $-1$.
7. Adding a scalar multiple of one row to another leaves the determinant **unchanged**.
8. For an upper or lower triangular matrix $T$:
   $$\det(T) = \prod_{i=1}^n T_{ii} = T_{11} T_{22} \cdots T_{nn}$$

#### Invertibility & The Matrix Inverse
A square matrix $A \in \mathbb{R}^{n\times n}$ is invertible (or non-singular) if and only if $\det(A) \neq 0$.
The inverse matrix $A^{-1}$ satisfies:
$$A A^{-1} = A^{-1} A = I_n$$

For a $2\times 2$ matrix $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:
$$A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

For general $n\times n$ matrices, the classical formula utilizes the **adjugate** matrix (matrix of transposed cofactors):
$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A) = \frac{1}{\det(A)} C^T$$
where cofactor $C_{ij} = (-1)^{i+j} M_{ij}$, and $M_{ij}$ is the determinant of the $(n-1)\times(n-1)$ submatrix obtained by deleting row $i$ and column $j$.

#### Uniqueness of Matrix Inverse (Proof)
Suppose $B$ and $C$ are both inverses of $A$. Then $AB = I$ and $CA = I$.
$$B = I B = (CA) B = C (AB) = C I = C$$
Thus, the inverse is uniquely defined. $\blacksquare$

---

### 2.2 Worked Example: $3\times 3$ Determinant and Adjugate Inversion

Let $A = \begin{pmatrix} 1 & 0 & 2 \\ 2 & -1 & 3 \\ 4 & 1 & 8 \end{pmatrix}$.

1. **Determinant via Laplace Cofactor Expansion (along Row 1):**
   $$\det(A) = 1 \cdot \begin{vmatrix} -1 & 3 \\ 1 & 8 \end{vmatrix} - 0 \cdot \begin{vmatrix} 2 & 3 \\ 4 & 8 \end{vmatrix} + 2 \cdot \begin{vmatrix} 2 & -1 \\ 4 & 1 \end{vmatrix}$$
   $$\det(A) = 1 [(-1)(8) - (3)(1)] - 0 + 2 [(2)(1) - (-1)(4)] = 1(-11) + 2(2 + 4) = -11 + 12 = 1$$
   Since $\det(A) = 1 \neq 0$, $A$ is invertible.

2. **Cofactor Matrix $C$:**
   - $C_{11} = + [(-1)(8) - (3)(1)] = -11$
   - $C_{12} = - [(2)(8) - (3)(4)] = -(16 - 12) = -4$
   - $C_{13} = + [(2)(1) - (-1)(4)] = 2 + 4 = 6$
   - $C_{21} = - [(0)(8) - (2)(1)] = -(-2) = 2$
   - $C_{22} = + [(1)(8) - (2)(4)] = 8 - 8 = 0$
   - $C_{23} = - [(1)(1) - (0)(4)] = -1$
   - $C_{31} = + [(0)(3) - (2)(-1)] = 2$
   - $C_{32} = - [(1)(3) - (2)(2)] = -(3 - 4) = 1$
   - $C_{33} = + [(1)(-1) - (0)(2)] = -1$

   $$C = \begin{pmatrix} -11 & -4 & 6 \\ 2 & 0 & -1 \\ 2 & 1 & -1 \end{pmatrix} \implies \text{adj}(A) = C^T = \begin{pmatrix} -11 & 2 & 2 \\ -4 & 0 & 1 \\ 6 & -1 & -1 \end{pmatrix}$$

3. **Inverse $A^{-1}$:**
   $$A^{-1} = \frac{1}{\det(A)} \text{adj}(A) = \begin{pmatrix} -11 & 2 & 2 \\ -4 & 0 & 1 \\ 6 & -1 & -1 \end{pmatrix}$$

4. **Sanity Check Multiplication:**
   $$A A^{-1} = \begin{pmatrix} 1 & 0 & 2 \\ 2 & -1 & 3 \\ 4 & 1 & 8 \end{pmatrix} \begin{pmatrix} -11 & 2 & 2 \\ -4 & 0 & 1 \\ 6 & -1 & -1 \end{pmatrix} = \begin{pmatrix} -11+0+12 & 2+0-2 & 2+0-2 \\ -22+4+18 & 4+0-3 & 4-1-3 \\ -44-4+48 & 8+0-8 & 8+1-8 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$ ✓

---

## 3. Vector Metrics and Geometry

### 3.1 Vector Norms, Angles, and Distance

A vector $x \in \mathbb{R}^n$ represents both a point in $n$-dimensional space and a directed line segment from the origin.

#### Vector $p$-Norms
A norm $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}_{\ge 0}$ assigns a non-negative length to every vector, satisfying:
1. **Positivity:** $\|x\| \ge 0$, and $\|x\| = 0 \iff x = 0$.
2. **Homogeneity:** $\|\alpha x\| = |\alpha| \|x\|$ for all $\alpha \in \mathbb{R}$.
3. **Triangle Inequality:** $\|x + y\| \le \|x\| + \|y\|$.

Common norms include:
- **$L_1$ Norm (Manhattan norm):** $\|x\|_1 = \sum_{i=1}^n |x_i|$. Promotes sparsity in machine learning (Lasso regression).
- **$L_2$ Norm (Euclidean norm):** $\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{x^T x}$. Standard geometric distance.
- **$L_\infty$ Norm (Max / Chebyshev norm):** $\|x\|_\infty = \max_{1 \le i \le n} |x_i|$.
- **General $L_p$ Norm ($p \ge 1$):** $\|x\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}$.

```
Visualizing Unit Balls in R^2 ({x : ||x|| <= 1}):
       L_1 (Diamond)           L_2 (Circle)         L_infinity (Square)
           y ^                     y ^                     y ^
             |                       |                       |
           1 +                     1 +                   1 +---+
            / \                    /   \                   |   |
      -1   /   \   1         -1   |     |   1        -1    |   |    1
     <----+-----+----> x    <-----+-----+-----> x   <------+---+------> x
           \   /                   \   /                   |   |
            \ /                     \ /                    |   |
          -1 +                    -1 +                  -1 +---+
             |                       |                       |
```

#### Dot Product, Angle, and Cauchy–Schwarz
The standard Euclidean dot product of $u, v \in \mathbb{R}^n$ is:
$$u \cdot v = u^T v = \sum_{i=1}^n u_i v_i$$
The angle $\theta \in [0, \pi]$ between $u$ and $v$ satisfies:
$$\cos\theta = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

#### Cauchy–Schwarz Inequality & Proof
**Theorem:** For all $u, v \in \mathbb{R}^n$, $|u \cdot v| \le \|u\|_2 \|v\|_2$.
*Proof:* For any real scalar $t \in \mathbb{R}$, consider the non-negative quadratic function:
$$q(t) = \|u - t v\|_2^2 = (u - t v) \cdot (u - t v) = \|u\|_2^2 - 2t (u \cdot v) + t^2 \|v\|_2^2 \ge 0$$
Since $q(t) \ge 0$ for all $t$, this quadratic polynomial has at most one real root. Hence, its discriminant $\Delta \le 0$:
$$\Delta = (-2(u \cdot v))^2 - 4(\|v\|_2^2)(\|u\|_2^2) = 4(u \cdot v)^2 - 4\|u\|_2^2 \|v\|_2^2 \le 0$$
$$(u \cdot v)^2 \le \|u\|_2^2 \|v\|_2^2 \implies |u \cdot v| \le \|u\|_2 \|v\|_2 \quad \blacksquare$$

---

## 4. Gauss Elimination and Systems of Linear Equations

### 4.1 Theory: System Characterization and Echelon Forms

A system of $m$ linear equations in $n$ variables is represented in matrix notation as:
$$Ax = b$$
where $A \in \mathbb{R}^{m\times n}$, $x \in \mathbb{R}^n$, and $b \in \mathbb{R}^m$.

The augmented matrix is defined as $[A \mid b] \in \mathbb{R}^{m\times(n+1)}$.

#### Elementary Row Operations
Three row operations preserve the exact solution set:
1. **$R_i \leftrightarrow R_j$:** Swap row $i$ and row $j$.
2. **$R_i \leftarrow c R_i$ ($c \neq 0$):** Scale row $i$ by a non-zero constant.
3. **$R_i \leftarrow R_i + c R_j$ ($i \neq j$):** Add a multiple of row $j$ to row $i$.

#### Row Echelon Form (REF) vs Reduced Row Echelon Form (RREF)
- **REF:** All non-zero rows are above zero rows. The leading entry (pivot) of a non-zero row is strictly to the right of the pivot in the row above. All entries below pivots are zero.
- **RREF:** In addition to REF conditions, every leading pivot is $1$, and every pivot is the *only* non-zero entry in its entire column.

```
       Row Echelon Form (REF)            Reduced Row Echelon Form (RREF)
       [ p  *  *  *  |  * ]                   [ 1  0  *  0  |  * ]
       [ 0  p  *  *  |  * ]                   [ 0  1  *  0  |  * ]
       [ 0  0  0  p  |  * ]                   [ 0  0  0  1  |  * ]
       [ 0  0  0  0  |  0 ]                   [ 0  0  0  0  |  0 ]
         (p = non-zero pivot)
```

#### Solvability Criterion (Rouché–Capelli Theorem)
- **Inconsistent (No solution):** $\text{rank}(A) < \text{rank}([A \mid b])$. Occurs when an augmented row reduces to $[0 \; 0 \; \cdots \; 0 \mid d]$ with $d \neq 0$.
- **Unique Solution:** $\text{rank}(A) = \text{rank}([A \mid b]) = n$ (number of unknowns, zero free variables).
- **Infinitely Many Solutions:** $\text{rank}(A) = \text{rank}([A \mid b]) = r < n$. The system has $n - r$ free variables.

---

### 4.2 Worked Example: Complete System Solution with Free Variables

Solve the linear system:
$$\begin{aligned}
x_1 + 2x_2 - x_3 + 3x_4 &= 2 \\
2x_1 + 4x_2 - x_3 + 8x_4 &= 7 \\
-x_1 - 2x_2 + 2x_3 - x_4 &= 1
\end{aligned}$$

1. **Construct Augmented Matrix and Perform Forward Elimination:**
   $$[A \mid b] = \left[\begin{array}{cccc|c} 1 & 2 & -1 & 3 & 2 \\ 2 & 4 & -1 & 8 & 7 \\ -1 & -2 & 2 & -1 & 1 \end{array}\right]$$
   
   Apply $R_2 \leftarrow R_2 - 2R_1$ and $R_3 \leftarrow R_3 + R_1$:
   $$\left[\begin{array}{cccc|c} 1 & 2 & -1 & 3 & 2 \\ 0 & 0 & 1 & 2 & 3 \\ 0 & 0 & 1 & 2 & 3 \end{array}\right]$$
   
   Apply $R_3 \leftarrow R_3 - R_2$:
   $$\left[\begin{array}{cccc|c} 1 & 2 & -1 & 3 & 2 \\ 0 & 0 & 1 & 2 & 3 \\ 0 & 0 & 0 & 0 & 0 \end{array}\right] \quad (\text{REF})$$

2. **Transform to RREF ($R_1 \leftarrow R_1 + R_2$):**
   $$\left[\begin{array}{cccc|c} 1 & 2 & 0 & 5 & 5 \\ 0 & 0 & 1 & 2 & 3 \\ 0 & 0 & 0 & 0 & 0 \end{array}\right] \quad (\text{RREF})$$

3. **Extract Pivots and Free Variables:**
   - Pivots are in Column 1 ($x_1$) and Column 3 ($x_3$).
   - Free variables are $x_2 = s$ and $x_4 = t$.
   
   Equations from RREF:
   $$x_1 + 2s + 5t = 5 \implies x_1 = 5 - 2s - 5t$$
   $$x_3 + 2t = 3 \implies x_3 = 3 - 2t$$

4. **Express General Solution in Vector Form ($x = x_p + x_h$):**
   $$x = \begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{pmatrix} = \begin{pmatrix} 5 \\ 0 \\ 3 \\ 0 \end{pmatrix} + s \begin{pmatrix} -2 \\ 1 \\ 0 \\ 0 \end{pmatrix} + t \begin{pmatrix} -5 \\ 0 \\ -2 \\ 1 \end{pmatrix}, \quad s, t \in \mathbb{R}$$
   Here, $x_p = (5, 0, 3, 0)^T$ is the particular solution, and the span of $\{(-2, 1, 0, 0)^T, (-5, 0, -2, 1)^T\}$ is the homogeneous solution space (the null space $N(A)$).

---

## 5. Vector Spaces and Subspaces

### 5.1 Vector Space Axioms & Subspace Criteria

A **vector space** $(V, +, \cdot)$ over $\mathbb{R}$ is a set $V$ equipped with vector addition and scalar multiplication satisfying the 8 axioms:
1. Associativity of addition: $u + (v + w) = (u + v) + w$.
2. Commutativity of addition: $u + v = v + u$.
3. Additive identity: $\exists 0 \in V$ such that $v + 0 = v$.
4. Additive inverse: $\forall v \in V, \exists (-v) \in V$ such that $v + (-v) = 0$.
5. Scalar compatibility: $a(bv) = (ab)v$.
6. Scalar identity: $1 \cdot v = v$.
7. Distributivity over vector addition: $a(u + v) = au + av$.
8. Distributivity over scalar addition: $(a + b)v = av + bv$.

#### The 3-Step Subspace Verification Test
A subset $W \subseteq V$ is a **subspace** of $V$ if and only if:
1. **Contains Zero Vector:** $0_V \in W$.
2. **Closed under Addition:** $\forall u, v \in W \implies u + v \in W$.
3. **Closed under Scalar Multiplication:** $\forall u \in W, \forall c \in \mathbb{R} \implies c u \in W$.

---

## 6. The Four Fundamental Subspaces

For any matrix $A \in \mathbb{R}^{m\times n}$, linear algebra reveals four canonical subspaces that govern the solvability of linear systems.

```
                  THE FUNDAMENTAL SUBSPACES OF A (m x n)
       Domain R^n                                 Codomain R^m
  +-----------------------+                  +-----------------------+
  |      Row Space        |                  |     Column Space      |
  |       C(A^T)          | ---- T_A(x) ---> |         C(A)          |
  |     dim = r = rank    |    (Bijective)   |      dim = r = rank   |
  +-----------------------+                  +-----------------------+
  |      Null Space       |                  |    Left Null Space    |
  |         N(A)          | ---- T_A(x) ---> |        N(A^T)         |
  |     dim = n - r       |      (= 0)       |      dim = m - r      |
  +-----------------------+                  +-----------------------+
    C(A^T) _|_ N(A) = R^n                      C(A) _|_ N(A^T) = R^m
```

| Subspace | Notation | Formal Definition | Living Space | Dimension |
|---|---|---|---|---|
| **Column Space** | $C(A)$ | $\{Ax \mid x \in \mathbb{R}^n\} = \text{span}(\text{cols of } A)$ | $\mathbb{R}^m$ | $r = \text{rank}(A)$ |
| **Null Space** | $N(A)$ | $\{x \in \mathbb{R}^n \mid Ax = 0\}$ | $\mathbb{R}^n$ | $n - r$ (Nullity) |
| **Row Space** | $C(A^T)$ | $\{A^T y \mid y \in \mathbb{R}^m\} = \text{span}(\text{rows of } A)$ | $\mathbb{R}^n$ | $r = \text{rank}(A)$ |
| **Left Null Space** | $N(A^T)$ | $\{y \in \mathbb{R}^m \mid A^T y = 0\}$ | $\mathbb{R}^m$ | $m - r$ |

#### Fundamental Theorem of Linear Algebra (Orthogonal Complements)
1. In $\mathbb{R}^n$: $N(A) = (C(A^T))^\perp$. The null space is the orthogonal complement of the row space.
2. In $\mathbb{R}^m$: $N(A^T) = (C(A))^\perp$. The left null space is the orthogonal complement of the column space.

#### Proof: $N(A) \perp C(A^T)$
Let $x \in N(A)$ and $v \in C(A^T)$.
Since $v \in C(A^T)$, $v = A^T y$ for some $y \in \mathbb{R}^m$.
Compute the inner product:
$$\langle v, x \rangle = v^T x = (A^T y)^T x = y^T (A x)$$
Since $x \in N(A)$, $Ax = 0$.
$$\langle v, x \rangle = y^T 0 = 0$$
Since this holds for every $v \in C(A^T)$ and every $x \in N(A)$, the two subspaces are strictly orthogonal. $\blacksquare$

---

## 7. Linear Independence, Basis, and Dimension

### 7.1 Definitions & Fundamental Theorems

- **Linear Independence:** A set of vectors $\{v_1, v_2, \dots, v_k\}$ is linearly independent if:
  $$c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0 \implies c_1 = c_2 = \cdots = c_k = 0$$
- **Spanning Set:** $\text{span}\{v_1, \dots, v_k\} = \{ \sum_{i=1}^k c_i v_i \mid c_i \in \mathbb{R} \}$.
- **Basis:** A set $\mathcal{B} = \{b_1, \dots, b_n\}$ is a basis of $V$ if it is **linearly independent** and **spans $V$**.
- **Dimension ($\dim V$):** The unique number of vectors in any basis of $V$.

#### Rank-Nullity Theorem (Dimension Theorem)
For any linear map represented by $A \in \mathbb{R}^{m\times n}$:
$$\text{rank}(A) + \text{nullity}(A) = n$$
$$\dim(C(A)) + \dim(N(A)) = n$$

---

## 8. Elementary Matrices and Inversion

### 8.1 Algebraic Representation of Row Operations

An **elementary matrix** $E$ is obtained by applying a single elementary row operation to $I_n$.
- Left-multiplying $A$ by $E$ applies the row operation to $A$: $A_{\text{new}} = E A$.
- Right-multiplying $A$ by $E$ applies the corresponding **column operation** to $A$.

```
Elementary Matrix Types:
1. Row Swap (R1 <-> R2):        2. Row Scale (R2 <- c*R2):     3. Row Add (R2 <- R2 - l*R1):
   E1 = [ 0  1  0 ]                E2 = [ 1  0  0 ]               E3 = [ 1  0  0 ]
        [ 1  0  0 ]                     [ 0  c  0 ]                    [-l  1  0 ]
        [ 0  0  1 ]                     [ 0  0  1 ]                    [ 0  0  1 ]
   Inv: E1^-1 = E1                 Inv: E2^-1 = diag(1,1/c,1)     Inv: E3^-1 = [ 1  0  0 ]
                                                                               [ l  1  0 ]
                                                                               [ 0  0  1 ]
```

#### Gauss-Jordan Inversion Algorithm
Form the block matrix $[A \mid I]$. Apply row operations $E_k \cdots E_1$ until $A$ becomes $I$:
$$(E_k \cdots E_1) [A \mid I] = [I \mid E_k \cdots E_1] = [I \mid A^{-1}]$$

---

## 9. LU and Cholesky Decompositions

### 9.1 LU Factorization Theory

Gauss elimination converts $A$ into an upper triangular matrix $U$ using elementary row additions:
$$E_k \cdots E_2 E_1 A = U \implies A = (E_1^{-1} E_2^{-1} \cdots E_k^{-1}) U = L U$$
where $L$ is a **unit lower triangular matrix** (1s on diagonal, multipliers $l_{ij}$ below diagonal), and $U$ is **upper triangular**.

#### Solving $Ax = b$ via LU Factorization in $O(n^2)$ Flops
Once $A = LU$ is computed ($O(n^3)$ once):
1. **Forward Substitution:** Solve $L y = b$ for $y$ ($O(n^2)$).
2. **Back Substitution:** Solve $U x = y$ for $x$ ($O(n^2)$).

```
Solving Ax = b via LU Decomposition:
         b
         |
         v
    [ L y = b ]  ---> Forward Substitution (O(n^2))
         |
         v
         y
         |
         v
    [ U x = y ]  ---> Back Substitution (O(n^2))
         |
         v
         x
```

#### LU with Partial Pivoting ($PA = LU$)
When diagonal pivots are zero or near-zero, row swaps are recorded in a permutation matrix $P$:
$$P A = L U$$

#### Cholesky Factorization for Symmetric Positive Definite Matrices
If $A \in \mathbb{R}^{n\times n}$ is **Symmetric Positive Definite (SPD)** ($A = A^T$ and $x^T A x > 0$ for all $x \neq 0$), then $A$ factors uniquely as:
$$A = L L^T$$
where $L$ is lower triangular with strictly positive diagonal entries. Cholesky factorization is twice as fast as LU ($n^3/3$ flops vs $2n^3/3$) and is unconditionally numerically stable without pivoting.

---

### 9.2 Worked Example: $3\times 3$ LU Decomposition

Factor $A = \begin{pmatrix} 2 & 1 & 1 \\ 4 & 5 & 4 \\ 6 & 9 & 14 \end{pmatrix}$ into $LU$.

1. **Elimination Step 1:**
   - Multiplier $l_{21} = \frac{4}{2} = 2$. $R_2 \leftarrow R_2 - 2R_1 \implies \text{Row 2} = (0, \; 5 - 2(1), \; 4 - 2(1)) = (0, 3, 2)$.
   - Multiplier $l_{31} = \frac{6}{2} = 3$. $R_3 \leftarrow R_3 - 3R_1 \implies \text{Row 3} = (0, \; 9 - 3(1), \; 14 - 3(1)) = (0, 6, 11)$.

2. **Elimination Step 2:**
   - Multiplier $l_{32} = \frac{6}{3} = 2$. $R_3 \leftarrow R_3 - 2R_2 \implies \text{Row 3} = (0, \; 6 - 2(3), \; 11 - 2(2)) = (0, 0, 7)$.

3. **Construct $L$ and $U$:**
   $$L = \begin{pmatrix} 1 & 0 & 0 \\ l_{21} & 1 & 0 \\ l_{31} & l_{32} & 1 \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 2 & 1 & 0 \\ 3 & 2 & 1 \end{pmatrix}, \qquad U = \begin{pmatrix} 2 & 1 & 1 \\ 0 & 3 & 2 \\ 0 & 0 & 7 \end{pmatrix}$$

4. **Solve $Ax = (4, 13, 37)^T$:**
   - Forward solve $Ly = (4, 13, 37)^T$:
     $$y_1 = 4$$
     $$2(4) + y_2 = 13 \implies y_2 = 5$$
     $$3(4) + 2(5) + y_3 = 37 \implies 12 + 10 + y_3 = 37 \implies y_3 = 15$$
     So $y = (4, 5, 15)^T$.
   - Back solve $Ux = (4, 5, 15)^T$:
     $$7x_3 = 15 \implies x_3 = \frac{15}{7}$$
     $$3x_2 + 2(15/7) = 5 \implies 3x_2 = 5 - \frac{30}{7} = \frac{5}{7} \implies x_2 = \frac{5}{21}$$
     $$2x_1 + \frac{5}{21} + \frac{15}{7} = 4 \implies 2x_1 + \frac{50}{21} = 4 \implies 2x_1 = \frac{34}{21} \implies x_1 = \frac{17}{21}$$

---

## 10. Change of Basis and Coordinate Transformations

### 10.1 Coordinate Vectors & Transition Matrices

Let $\mathcal{B} = \{b_1, \dots, b_n\}$ be an ordered basis of $\mathbb{R}^n$. Every vector $v \in \mathbb{R}^n$ has a unique representation:
$$v = c_1 b_1 + c_2 b_2 + \cdots + c_n b_n$$
The coordinate vector of $v$ relative to $\mathcal{B}$ is $[v]_\mathcal{B} = (c_1, c_2, \dots, c_n)^T$.

Let $P_\mathcal{B} = [b_1 \; b_2 \; \cdots \; b_n]$. Then:
$$v = P_\mathcal{B} [v]_\mathcal{B} \iff [v]_\mathcal{B} = P_\mathcal{B}^{-1} v$$

#### Transition Matrix Between Two Bases
To convert coordinates from basis $\mathcal{B}$ to basis $\mathcal{C}$:
$$[v]_\mathcal{C} = P_{\mathcal{C} \leftarrow \mathcal{B}} [v]_\mathcal{B}, \qquad \text{where } P_{\mathcal{C} \leftarrow \mathcal{B}} = P_\mathcal{C}^{-1} P_\mathcal{B}$$

#### Transformation Matrix Under Change of Basis (Similarity Transformation)
If a linear transformation $T: V \to V$ has matrix $[T]_\mathcal{B}$ in basis $\mathcal{B}$, its matrix in basis $\mathcal{C}$ is:
$$[T]_\mathcal{C} = P^{-1} [T]_\mathcal{B} P, \quad \text{where } P = P_{\mathcal{B} \leftarrow \mathcal{C}}$$
Matrices $A$ and $B$ are **similar** ($A \sim B$) if $B = P^{-1} A P$. Similar matrices share the same determinant, trace, eigenvalues, and characteristic polynomial.

---

## 11. Inner Product Spaces and Orthogonality

### 11.1 Axiomatic Inner Products

An inner product on a real vector space $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:
1. **Symmetry:** $\langle u, v \rangle = \langle v, u \rangle$.
2. **Linearity in First Argument:** $\langle a u + b v, w \rangle = a\langle u, w \rangle + b\langle v, w \rangle$.
3. **Positive-Definiteness:** $\langle v, v \rangle \ge 0$, and $\langle v, v \rangle = 0 \iff v = 0$.

Induced norm: $\|v\| = \sqrt{\langle v, v \rangle}$.

#### Examples of Inner Product Spaces
- **Euclidean $\mathbb{R}^n$:** $\langle u, v \rangle = u^T v$.
- **Weighted $\mathbb{R}^n$:** $\langle u, v \rangle = u^T W v$, where $W$ is symmetric positive definite.
- **Function Space $C[a, b]$:** $\langle f, g \rangle = \int_a^b f(x) g(x) dx$. (Foundation of Fourier Series).

#### Orthogonal & Orthonormal Sets
A set $\{q_1, q_2, \dots, q_k\}$ is **orthonormal** if:
$$\langle q_i, q_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

#### Generalized Fourier Expansion Theorem
If $\{q_1, \dots, q_n\}$ is an orthonormal basis for $V$, then every $v \in V$ can be decomposed trivially without matrix inversion:
$$v = \sum_{i=1}^n \langle v, q_i \rangle q_i$$

---

## 12. The Gram–Schmidt Orthogonalization Process

### 12.1 Classical vs Modified Gram–Schmidt

Given a linearly independent set $\{v_1, v_2, \dots, v_k\}$, Gram–Schmidt constructs an orthogonal basis $\{u_1, u_2, \dots, u_k\}$:

$$\begin{aligned}
u_1 &= v_1 \\
u_2 &= v_2 - \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1 \\
u_3 &= v_3 - \frac{\langle v_3, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1 - \frac{\langle v_3, u_2 \rangle}{\langle u_2, u_2 \rangle} u_2 \\
&\;\;\vdots \\
u_i &= v_i - \sum_{j=1}^{i-1} \frac{\langle v_i, u_j \rangle}{\langle u_j, u_j \rangle} u_j
\end{aligned}$$

Normalize to obtain orthonormal basis: $q_i = \frac{u_i}{\|u_i\|}$.

```
Gram-Schmidt Geometric Step (u_2 = v_2 - proj_{u_1}(v_2)):
                 v_2 ^
                     | \
                     |   \  u_2 = v_2 - proj_{u_1}(v_2)  (Orthogonal to u_1!)
                     |     \
                     +-------> u_1 (= v_1)
                     proj_{u_1}(v_2)
```

#### Classical (CGS) vs Modified Gram–Schmidt (MGS)
In floating-point arithmetic, CGS suffers from catastrophic loss of orthogonality due to cancellation. **Modified Gram–Schmidt (MGS)** subtracts projections sequentially from intermediate vectors, maintaining numerical stability:
For $i = 1, \dots, k$:
1. $q_i \leftarrow v_i / \|v_i\|$
2. For $j = i+1, \dots, k$:
   $v_j \leftarrow v_j - \langle v_j, q_i \rangle q_i$

---

## 13. Orthogonal Projection onto a Subspace

### 13.1 Projection Operator & Matrix Derivation

Let $W \subset \mathbb{R}^m$ be a subspace with basis columns forming matrix $A \in \mathbb{R}^{m\times n}$ (so $W = C(A)$).
For any vector $b \in \mathbb{R}^m$, the orthogonal projection $p = \text{proj}_W(b)$ satisfies:
1. $p \in C(A) \implies p = A \hat{x}$ for some $\hat{x} \in \mathbb{R}^n$.
2. The error (residual) vector $e = b - p = b - A\hat{x}$ is orthogonal to $W = C(A)$.
   $$A^T e = 0 \implies A^T (b - A\hat{x}) = 0 \implies A^T A \hat{x} = A^T b$$

If $A$ has linearly independent columns, $A^T A$ is invertible, yielding:
$$\hat{x} = (A^T A)^{-1} A^T b$$
$$p = A \hat{x} = A (A^T A)^{-1} A^T b = P b$$

where the **orthogonal projection matrix** onto $C(A)$ is:
$$P = A (A^T A)^{-1} A^T$$

```
Orthogonal Projection Geometry:
              b ^
                | \
                |   \  e = b - Pb  (Orthogonal to Subspace W)
                |     \
              0 +======p=======> W = C(A)
                   p = Pb = A(A^T A)^-1 A^T b
```

#### Fundamental Properties of Projection Matrices
1. **Idempotence:** $P^2 = P$ (projecting an already-projected vector changes nothing).
2. **Symmetry:** $P^T = P$.
3. **Eigenvalues:** $\lambda \in \{0, 1\}$.
4. **Complementary Projector:** $I - P$ is the orthogonal projector onto $W^\perp = N(A^T)$.

---

## 14. The Least-Squares Principle and Regression

### 14.1 Problem Formulation and Normal Equations

When a system $Ax = b$ is overdetermined ($m > n$, more equations than unknowns) and inconsistent ($b \notin C(A)$), no exact solution exists.
The **least-squares problem** seeks $\hat{x}$ minimizing the squared Euclidean norm of the residual:
$$\min_{x \in \mathbb{R}^n} \|Ax - b\|_2^2 = \min_{x \in \mathbb{R}^n} \sum_{i=1}^m \left( \sum_{j=1}^n A_{ij} x_j - b_i \right)^2$$

From Section 13, the optimal vector $A\hat{x}$ must be the closest point in $C(A)$ to $b$, which is the orthogonal projection $Pb$. Hence $\hat{x}$ satisfies the **Normal Equations**:
$$A^T A \hat{x} = A^T b$$

#### Calculus Derivation of Normal Equations
Define the objective function $S(x) = \|Ax - b\|_2^2 = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b$.
Taking the gradient with respect to $x$ and setting to zero:
$$\nabla_x S(x) = 2 A^T A x - 2 A^T b = 0 \implies A^T A x = A^T b \quad \blacksquare$$

---

### 14.2 Worked Example: Polynomial / Linear Least-Squares Curve Fitting

Fit a best-fit line $y = c_0 + c_1 x$ through the data points $(1, 1), (2, 2), (3, 4)$.

1. **Set Up System $A c = b$:**
   $$\begin{aligned}
   c_0 + 1 c_1 &= 1 \\
   c_0 + 2 c_1 &= 2 \\
   c_0 + 3 c_1 &= 4
   \end{aligned} \implies A = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}, \quad b = \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix}, \quad c = \begin{pmatrix} c_0 \\ c_1 \end{pmatrix}$$

2. **Compute $A^T A$ and $A^T b$:**
   $$A^T A = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}$$
   $$A^T b = \begin{pmatrix} 1 & 1 & 1 \\ 1 & 2 & 3 \end{pmatrix} \begin{pmatrix} 1 \\ 2 \\ 4 \end{pmatrix} = \begin{pmatrix} 1+2+4 \\ 1+4+12 \end{pmatrix} = \begin{pmatrix} 7 \\ 17 \end{pmatrix}$$

3. **Solve Normal Equations $\begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix} \begin{pmatrix} c_0 \\ c_1 \end{pmatrix} = \begin{pmatrix} 7 \\ 17 \end{pmatrix}$:**
   $$\det(A^T A) = (3)(14) - (6)(6) = 42 - 36 = 6$$
   $$(A^T A)^{-1} = \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix}$$
   $$\begin{pmatrix} c_0 \\ c_1 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \begin{pmatrix} 7 \\ 17 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} 98 - 102 \\ -42 + 51 \end{pmatrix} = \frac{1}{6} \begin{pmatrix} -4 \\ 9 \end{pmatrix} = \begin{pmatrix} -2/3 \\ 3/2 \end{pmatrix}$$

   Best-fit line: $y = -\frac{2}{3} + \frac{3}{2} x = -0.667 + 1.5 x$.

---

## 15. QR Decomposition

### 15.1 Factorization & Computational Advantage

Every matrix $A \in \mathbb{R}^{m\times n}$ ($m \ge n$) with linearly independent columns can be factored as:
$$A = Q R$$
- **$Q \in \mathbb{R}^{m\times n}$** has orthonormal columns: $Q^T Q = I_n$.
- **$R \in \mathbb{R}^{n\times n}$** is upper triangular and invertible with positive diagonal entries.

```
QR Decomposition Architecture:
      A (m x n)     =        Q (m x n)        *      R (n x n)
  [ |   |       | ]       [ |   |       | ]       [ r_11  r_12  ... r_1n ]
  [ a_1 a_2 ... a_n]  =   [ q_1 q_2 ... q_n]  *   [  0   r_22  ... r_2n ]
  [ |   |       | ]       [ |   |       | ]       [  0    0    ... r_nn ]
                          (Orthonormal Cols)      (Upper Triangular)
```

#### Solving Least Squares via QR Decomposition
Substitute $A = QR$ into the Normal Equations $A^T A \hat{x} = A^T b$:
$$(QR)^T (QR) \hat{x} = (QR)^T b \implies R^T \underbrace{(Q^T Q)}_{= I} R \hat{x} = R^T Q^T b \implies R^T R \hat{x} = R^T Q^T b$$
Since $R$ is invertible, $R^T$ is invertible. Canceling $R^T$ yields:
$$R \hat{x} = Q^T b$$

#### Why QR is Superior to Normal Equations
Forming $A^T A$ explicitly squares the condition number: $\kappa(A^T A) = (\kappa(A))^2$. If $\kappa(A) = 10^4$, then $\kappa(A^T A) = 10^8$, losing 8 digits of precision! QR operates directly on $A$, maintaining $\kappa(R) = \kappa(A)$ and preventing catastrophic loss of precision.

---

## 16. Comprehensive Solved Exam-Style Problems

### Problem 1: Complete $4\times 4$ Determinant by Row Reduction
**Statement:** Compute $\det(A)$ where $A = \begin{pmatrix} 1 & 2 & 3 & 1 \\ 2 & 5 & 8 & 3 \\ 1 & 2 & 5 & 4 \\ 3 & 6 & 9 & 7 \end{pmatrix}$.

**Step-by-step Solution:**
Apply row operations (which leave determinant unchanged):
- $R_2 \leftarrow R_2 - 2R_1 \implies (0, 1, 2, 1)$
- $R_3 \leftarrow R_3 - R_1 \implies (0, 0, 2, 3)$
- $R_4 \leftarrow R_4 - 3R_1 \implies (0, 0, 0, 4)$

Resulting Upper Triangular Matrix:
$$U = \begin{pmatrix} 1 & 2 & 3 & 1 \\ 0 & 1 & 2 & 1 \\ 0 & 0 & 2 & 3 \\ 0 & 0 & 0 & 4 \end{pmatrix}$$
$$\det(A) = \det(U) = (1)(1)(2)(4) = \mathbf{8}$$

---

### Problem 2: Basis and Dimension of the Four Subspaces
**Statement:** Find bases and dimensions of $C(A), N(A), C(A^T), N(A^T)$ for $A = \begin{pmatrix} 1 & 3 & 3 & 2 \\ 2 & 6 & 9 & 7 \\ -1 & -3 & 3 & 4 \end{pmatrix}$.

**Step-by-step Solution:**
Row reduce $[A]$:
$$\begin{pmatrix} 1 & 3 & 3 & 2 \\ 2 & 6 & 9 & 7 \\ -1 & -3 & 3 & 4 \end{pmatrix} \xrightarrow[R_3+R_1]{R_2-2R_1} \begin{pmatrix} 1 & 3 & 3 & 2 \\ 0 & 0 & 3 & 3 \\ 0 & 0 & 6 & 6 \end{pmatrix} \xrightarrow{R_3-2R_2} \begin{pmatrix} 1 & 3 & 3 & 2 \\ 0 & 0 & 3 & 3 \\ 0 & 0 & 0 & 0 \end{pmatrix} \xrightarrow{R_1-R_2,\; R_2/3} \begin{pmatrix} 1 & 3 & 0 & -1 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$
- $\text{rank}(A) = 2$.
1. **$C(A)$:** Pivot columns of original $A$ are columns 1 and 3:
   $$\text{Basis}(C(A)) = \left\{ \begin{pmatrix} 1 \\ 2 \\ -1 \end{pmatrix}, \begin{pmatrix} 3 \\ 9 \\ 3 \end{pmatrix} \right\}, \quad \dim(C(A)) = 2$$
2. **$C(A^T)$:** Non-zero rows of RREF:
   $$\text{Basis}(C(A^T)) = \left\{ \begin{pmatrix} 1 \\ 3 \\ 0 \\ -1 \end{pmatrix}, \begin{pmatrix} 0 \\ 0 \\ 1 \\ 1 \end{pmatrix} \right\}, \quad \dim(C(A^T)) = 2$$
3. **$N(A)$:** Free variables $x_2 = s, x_4 = t \implies x_1 = -3s + t, x_3 = -t$:
   $$\text{Basis}(N(A)) = \left\{ \begin{pmatrix} -3 \\ 1 \\ 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 \\ 0 \\ -1 \\ 1 \end{pmatrix} \right\}, \quad \dim(N(A)) = 4 - 2 = 2$$
4. **$N(A^T)$:** Left null space, $m - r = 3 - 2 = 1$. Row reduce $[A^T \mid 0]$:
   $$\text{Basis}(N(A^T)) = \left\{ \begin{pmatrix} 5 \\ -2 \\ 1 \end{pmatrix} \right\}, \quad \dim(N(A^T)) = 1$$

---

### Problem 3: Complete Gram–Schmidt and QR Factorization
**Statement:** Find the QR decomposition of $A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$.

**Step-by-step Solution:**
Let $a_1 = (1, 1, 0)^T$ and $a_2 = (1, 0, 1)^T$.
1. **First Vector:**
   $$u_1 = a_1 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}, \quad \|u_1\|_2 = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}$$
   $$q_1 = \frac{u_1}{\|u_1\|_2} = \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{pmatrix}$$
   $$r_{11} = \|u_1\|_2 = \sqrt{2}$$

2. **Second Vector:**
   $$r_{12} = q_1^T a_2 = \left( \frac{1}{\sqrt{2}} \right)(1) + \left( \frac{1}{\sqrt{2}} \right)(0) + (0)(1) = \frac{1}{\sqrt{2}}$$
   $$u_2 = a_2 - r_{12} q_1 = \begin{pmatrix} 1 \\ 0 \\ 1 \end{pmatrix} - \frac{1}{\sqrt{2}} \begin{pmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{pmatrix} = \begin{pmatrix} 1 - 1/2 \\ 0 - 1/2 \\ 1 - 0 \end{pmatrix} = \begin{pmatrix} 1/2 \\ -1/2 \\ 1 \end{pmatrix}$$
   $$r_{22} = \|u_2\|_2 = \sqrt{(1/2)^2 + (-1/2)^2 + 1^2} = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2} = \frac{\sqrt{6}}{2}$$
   $$q_2 = \frac{u_2}{r_{22}} = \frac{1}{\sqrt{3/2}} \begin{pmatrix} 1/2 \\ -1/2 \\ 1 \end{pmatrix} = \begin{pmatrix} 1/\sqrt{6} \\ -1/\sqrt{6} \\ 2/\sqrt{6} \end{pmatrix}$$

3. **Assemble Matrices:**
   $$Q = \begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{6} \\ 1/\sqrt{2} & -1/\sqrt{6} \\ 0 & 2/\sqrt{6} \end{pmatrix}, \qquad R = \begin{pmatrix} \sqrt{2} & 1/\sqrt{2} \\ 0 & \sqrt{3/2} \end{pmatrix}$$

---

## 17. Unit I Summary & Formula Cheat Sheet

| Mathematical Concept | Defining Formula / Key Theorem | Computational Purpose |
|---|---|---|
| **Matrix Multiplication** | $(AB)_{ij} = \sum_k A_{ik} B_{kj}$ | Composes linear maps; associative, non-commutative |
| **Determinant** | $\det(AB) = \det(A)\det(B)$ | Volume scaling factor; zero $\iff$ singular |
| **Inverse Formula** | $A^{-1} = \frac{1}{\det(A)}\text{adj}(A)$ | Explicit analytic inverse; $O(n^3)$ via Gauss–Jordan |
| **Rank-Nullity Theorem** | $\text{rank}(A) + \text{nullity}(A) = n$ | Total column dimensions conserved |
| **Fundamental Subspaces** | $N(A) = C(A^T)^\perp, \; N(A^T) = C(A)^\perp$ | Complete geometric decomposition of $\mathbb{R}^n, \mathbb{R}^m$ |
| **LU Decomposition** | $A = LU \implies Ly=b, \; Ux=y$ | Solves linear systems in $O(n^2)$ after $O(n^3)$ factoring |
| **Projection Matrix** | $P = A(A^T A)^{-1} A^T$ | Projects onto $C(A)$; $P^2=P, \; P^T=P$ |
| **Normal Equations** | $A^T A \hat{x} = A^T b$ | Finds unique least-squares fit minimizing $\Vert Ax-b \Vert_2^2$ |
| **QR Decomposition** | $A = QR \implies R\hat{x} = Q^T b$ | Numerically stable least squares; condition number preserved |
