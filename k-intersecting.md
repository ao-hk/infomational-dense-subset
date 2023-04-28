---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.8
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"tags": []}

# Dense Subsets

```{code-cell} ipython3
:tags: []

from visualization import playground
```

```{important}
How to identify dense subsets that are meaningful clusters?
```

+++ {"tags": []}

## Densest at-least-$k$-subsets

+++

Consider the following extension of the densest $k$-subsets problem in {eq}`densest`:

+++

---

**Definition** (densest at-least-$k$ {cite}`andersen2009finding, khuller2009finding`)  
:label: def:densest-at-least-k

For order $k\in \Set{1,\dots, n}$, define

$$
\begin{align}
\mc{D}_{\geq k} &:= \arg \overbrace{\max_{\substack{B \subseteq V:\\ \abs{B}\geq k}}\rho(B)}^{\rho_{\geq k}:=},
\end{align}
$$ (densest-at-least-k)

which is the set of densest subsets with at least $k$ nodes.

---

+++

$\mc{D}_{\geq k}$ is meaningful because
- $\mc{D}_{\geq 1}=\mc{D}_*$ is the set of non-empty densest subsets and
- $\mc{D}_{\geq k} \subseteq \bigcup_{\ell=k}^n \mc{D}_k$ even though $\mc{D}_k\subseteq \mc{D}_{\geq k}$ may not hold.

+++

```{important}
- Is it enough to look at $\mc{D}_{\geq k}$?
- How to compute $\mc{D}_{\geq k}$?
```

+++

For $k\in \Set{1,\dots, n}$,
$$
\begin{alignat}{2}
&&x &= \rho_k\\
&\stackrel{\text{(a)}}\iff& x &= \max_{\substack{B \subseteq V:\\ \abs{B}\geq k}} \rho(B)\\
&\stackrel{\text{(b)}}\iff& 0 &= \max_{\substack{B \subseteq V:\\ \abs{B}\geq k}} \frac{\mu(B)}{\nu(B)} - x\\
&\stackrel{\text{(c)}}\iff& 0 &= \max_{\substack{B \subseteq V:\\ \abs{B}\geq k}} \left[\mu(B) - x\nu(B)\right]\\
\end{alignat}
$$

and the set of optimal subsets $B$ remain unchanged.

- (a) is by the definition {eq}`densest-at-least-k` of $\rho_k$;
- (b) is by subtracting $x$ from both sides; and
- &#40;c) is by multiplying both sides by $\nu(B)\geq 0$.

+++

```{important}
A supermodular function can be maximized over a lattice of subsets of order $n$ in a polynomial number $\text{SFM}(n)$ of oracle calls to evaluate the function {cite}`iwata2001combinatorial`.

Since
- $\mu(B) - x\nu(B)$ is supermodular in $B$ for $x\geq 0$ and
- the root is non-negative, i.e., $x=\rho_k\geq 0$,

can the root $\rho_k$ be computed in polynomial time?
```

+++

```{caution}
The maximization is over $\Set{B\subseteq V\mid \abs{B}\geq k}$, which is not a lattice family unless $k=0$.
```

+++ {"tags": []}

## $k$-intersecting dense subsets

+++

$\Set{B\subseteq V \mid \abs{B}\geq k}$ is the so-called *$k$-intersecting family* {cite}`schrijver2002combinatorial`:

+++

A family $\mc{F}$ is *$k$-intersecting* if

$$
\begin{align}
B_1\cap B_2, B_1 \cup B_2 \in  \mc{F} \qquad \forall B_1, B_2 \in \mc{F}: \abs{B_1\cap B_2}\geq k.
\end{align}
$$ (k-intersecting)

If the above holds for $k=0$, $\mc{F}$ is called a *lattice family*.

+++

```{important}
Maximizing $\mu(B)-x\nu(B)$ over a $k$-intersecting family can be reduced to multiple maximizations over different lattice families.
```

+++

---

**Definition** ($k$-intersecting dense)  
:label: def:k-intersecting-dense

For order $k\in \Set{0,\dots, n}$, define

$$
\begin{align}
\mc{U}_k(x) &:= \arg \underbrace{\overbrace{\max_{\substack{B\subseteq V:\\ \abs{B}\geq k}} \underbrace{\left[\mu(B) - x\nu(B)\right]}_{f[B](x):=}}^{f_k(x):=}}_{\displaystyle =\max_{\substack{A\subseteq V:\\ \abs{A}= k}} \underbrace{\max_{\substack{B\subseteq V:\\ B\supseteq A}} f[B](x)}_{f_A(x):=}},
\end{align}
$$ (k-intersecting-dense)

where $\mu$ and $\nu$ are respectively the mass and volume function for the density defined in {prf:ref}`def:density`.

---

```{code-cell} ipython3
:tags: []

playground(b=4, k=1, m=1)
```

It follows that 
- $\rho_k$ is a root of $f_k(x)=0$ and 
- $\mc{D}_{\geq k}$ is the corresponding set $\mc{U}_k(x)$ of maximizing subsets $B$ at the root.

+++

---

**Theorem** (root of $f_k$)   
:label: thm:root-of-f-k

The set of roots of $f_k$ defined in {eq}`k-intersecting-dense` is

$$
\begin{align}
\Set{x \in \mathbb{R} |f_k(x)= 0} &= 
\begin{cases}
\Set{\rho_k} & k\in \Set{1,\dots,n}\\
[\rho_1,\infty) & k=0
\end{cases}
\end{align}
$$ (sol:rho-k)

and the corresponding set $\mc{U}_k(x)$ of maximizing subsets gives

$$
\begin{align}
\mc{U}_k(\rho_k) &= 
\begin{cases}
\mc{D}_{\geq k} & k\in \Set{1,\dots,n}\\
\mc{D}_{\geq 1} \cup \Set{\emptyset} & k=0.
\end{cases}
\end{align}
$$ (sol:U-k-at-rho-k)

---

+++

For any $x\geq 0$, 

$$f_A(x):=\max_{\substack{B\subseteq V:\\ B\supseteq A}} f[B](x)$$ 

in {eq}`k-intersecting-dense` can be solve in $\text{SFM}(n-k)$ oracle calls because

- $f[B](x):=\mu(B) - x\nu(B)$ is supermodular for $x\geq 0$ and
- $\Set{B\subseteq V\mid B\supseteq A}$ is a lattice family.

+++

Furthermore:
- The maximization can be solved for all $x$ with an additional factor $(n-k)$ of oracle calls.
- The set of maximizing subsets $B$ form a lattice family, which can be computed with an addition factor $(n-k)$ of oracle calls.

+++

---

**Theorem** (parametric SFM)  
:label: thm:parametric-SFM


For $x\in \mathbb{R}$, $\mc{U}_k(x)$ is a $k$-intersecting family {eq}`k-intersecting` satisfying

$$
\begin{align}
\mc{U}_k(x) &\supseteq 
\begin{cases}
V & x < 0\\
\emptyset & k=0, x> \xi_0 \\
\mc{D}_k & k\in \Set{1,\dots,n}, x> \xi_k 
\end{cases}
\end{align}
$$

for some $\xi_k\geq 0$ possibly dependent on $k\in \Set{0,\dots, n}$, where equality holds if $h$ is strictly increasing. 

Furthermore, the function $\mc{U}_k$ of $x$ can be computed in $\mc{O}\left({n \choose k} (n-k)^2 \right)$ submodular function minimizations of order $n-k$.

---

+++

```{important}
The number of oracle calls to compute $\mc{U}_k(x)$ is polynomial for sufficiently small order $k\in \mc{O}(\log n)$.
- Is it enough to consider small orders?
- Is it enough to consider $\mc{U}_k(\rho_k)$? 
```

+++ {"tags": []}

## Beyond maximal dense subsets

+++

As an example, consider the usual graph density in {eq}`rho:graph`.

+++

For the dumbbell graph in {prf:ref}`eg:dumbbell`, it can be shown that $V$ is the unique densest subset and so

$$
\begin{align}
\mc{D}_{\geq k} &=
\Set{V} \qquad \forall k\in \Set{1,\dots,n},
\end{align}
$$

which fails to identify the $b$-cliques as clusters.

+++

$\mc{U}_k(\rho_k)$ fails as well because it is essentially $\mc{D}_{\geq k}$ by {prf:ref}`thm:root-of-f-k`.

+++

```{important}
Is it enough to consider $\mc{U}_0(x)$ for different $x$?
```

```{code-cell} ipython3
:tags: []

playground(b=4, k=0, m=0, t="broken")
```

Note that $$f[B](x):=\mu(B) - x\nu(B)$$

- $y$-intercept: $\mu(B)\geq 0$
- slope: $-\nu(B)$.
- $x$-intercept: $\rho(B)=\frac{\mu(B)}{\nu(B)}\geq 0$


$$f_0(x):=\max_{B\subseteq V} f[B](x)$$ is

- non-negative because $f[\emptyset](x)=0$ for all $x$; 
- non-increasing because $f[B](x)$ has non-positive slope $-\nu(B)\leq 0$; and 
- concave because of the maximization over linear $f[B]$.

This implies the following structure of $\mc{U}_0(x)$.

+++

---

**Theorem** ($\nu$-maximally-dense)  
:label: thm:nu-maximally-dense

For $x\in \mathbb{R}$, $\mc{U}_0(x)$ is *$\nu$-maximally dense* in the sense that for all $B\in \mc{U}_0(x)\setminus\Set{\emptyset}$,

$$
\begin{align}
\rho(B)> \rho(C) \qquad \forall C\subseteq V: \nu(C)>\nu(B)\\
\rho(B)\geq \rho(C) \qquad \forall C\subseteq V: \nu(C)=\nu(B).
\end{align}
$$ (nu-maximally-dense)

---

+++

For the dumbbell graph in {prf:ref}`eg:dumbbell`, since $V$ has the largest volume as well, {eq}`nu-maximally-dense` implies

$$
\begin{align}
\bigcup_{x\in \mathbb{R}} \mc{U}_{0}(x) &=
\Set{V},
\end{align}
$$

which again fails to identify the $b$-cliques as clusters.

+++

For the dumbbell graph in {prf:ref}`eg:dumbbell` with $b=3$,

$$
\begin{align}
\mc{U}_{1}(x) &=
\begin{cases}
\Set{V} && x\leq \frac43\\
\Set{C_1,C_2} && x\in [\frac43,\frac32]\\
\Set{i} && i\in V, x\geq \frac32
\end{cases}
\end{align}
$$

which successfully identifies the $b$-cliques on $C_1$ and $C_2$.

```{code-cell} ipython3
:tags: []

playground(b=3, k=1, m=0, t="normal")
```

```{code-cell} ipython3

```
