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

# Strong Subsets

```{code-cell} ipython3
:tags: []

from visualization import playground

playground(b=4, k=1, m=0, t="broken")
```

For the biased dumbbell graph in {prf:ref}`eg:broken-dumbbell`, the ring on $C_2$ cannot be identified by $\mc{U}_k(x)$. This is because the clique $C_1$ dominates $C_2$ in the sense that

$$
\underbrace{f[C_1](x)}_{\mu(C_1)-x\nu(C_1)} > \underbrace{f[C_2](x)}_{\mu(C_2)-x\nu(C_2)} \qquad \forall x\geq 0.
$$

as

- $\mu(C_1)={b \choose 2}>b=\mu(C_2)$, and 
- $\nu(C_1)=b=\nu(C_2)$.

+++

```{important}
How to find locally dense subsets that may not be globally dense?
```

+++ {"tags": []}

## Strength

+++

Instead of comparing subsets locally by their densities, we compare them using the following measure of strength:

+++

---

**Definition** (strength)  
:label: def:strength

For order $k\in \Set{0,\dots, n}$ and subset $C\subseteq V$, define the strength
$$
\sigma_k(C):= \min_{\substack{B\subseteq C:\nu(C|B)>0\\ \abs{B}\geq k}} \frac{\mu(C|B)}{\nu(C|B)},
$$ (strength)

where $\mu$ and $\nu$ are respectively the mass and volume function for the density defined in {prf:ref}`def:density`, and the conditional versions are as defined in {eq}`conditional`. 

As a convention the strength is $\infty$ for $k\geq \abs{C}$ where the minimization has no feasible solution.

---

+++

The constraint $\nu(C|B)>0$ ensures that the strength is well-defined. When $\nu(B)=\abs{B}$, the constraint is equivalent to $B\subsetneq C$ and so the strength reduces to

$$
\sigma_k(C) = \min_{\substack{B\subsetneq C:\\ \abs{B}\geq k}} \frac{\mu(C|B)}{\abs{C\setminus B}},
$$ (volume=cardinality:strength)

+++

The following definition of strong subsets further confines the comparisons of strong subsets to local instead of global comparisons:

+++

---

**Definition** (strong subsets)  
:label: def:strong-subsets


For order $k\in \Set{0,\dots, n}$ and a strength function $\sigma$, define for all $x\in \mathbb{R}$

$$
\begin{align}
\mc{C}_{\sigma}(x) &:= \operatorname{maximal} \Set{C\subseteq V| \sigma(C)>x},
\end{align}
$$ (maximal-strong-subsets)

where 

$$
\begin{align}
\operatorname{maximal} \mc{F} &:= \Set{B\in \mc{F}|\not\exists C\in \mc{F},B\subsetneq C}.
\end{align}
$$ (maximal)

---

+++

For the broken dumbbell graph in {prf:ref}`eg:broken-dumbbell`, the following shows that $C_2\in \mc{C}_{\sigma_1}$, i.e.,

$$\sigma_1(C) < \sigma_1(C_2) \qquad \forall C\subseteq V:C\supsetneq C_2$$

and so $C_2$ can be identified as a maximal strong subset.

+++

$$
\begin{align}
\sigma_1(C_2) &= \min_{\substack{B\subsetneq C_2:\\ \abs{B}\geq 1}} \frac{\mu(C_2|B)}{\abs{C_2\setminus B}}\\
&=  \frac{\mu(C_2|\Set{i})}{\abs{C_2\setminus \Set{i}}} && \forall i\in C_2\\
&= \frac{{b \choose 2} - 1}{b-1} = \frac{b}{2} - \frac1{b-1}
\end{align}
$$

+++

$$
\begin{align}
\sigma_1(V) &= \min_{\substack{B\subsetneq V:\\ \abs{B}\geq 1}} \frac{\mu(V|B)}{\abs{V\setminus B}}\\
&=  \frac{\mu(V|C_1)}{\abs{V\setminus C_1}}\\
&= \frac{{b \choose 2} - 1+1}{b} = \frac{b}{2} - \frac12 < \sigma_1(C_2)
\end{align}
$$

+++

For all $C\subsetneq V: C\supsetneq C_2$, 

$$
\begin{align}
\sigma_1(C) &= \min_{\substack{B\subsetneq C:\\ \abs{B}\geq 1}} \frac{\mu(C|B)}{\abs{C\setminus B}}\\
&\leq  \frac{\mu(C|C_2)}{\abs{C\setminus C_2}}\\
&= \frac{1+{\abs{C}-b\choose 2}}{\abs{C}-b}= \frac1{\abs{C}-b} + \frac{\abs{C}-b-1}{2}< \sigma_1(C_2)
\end{align}
$$

+++ {"tags": []}

## Characterization and computation

+++

---

**Definition** (restricted $k$-intersecting dense)  
:label: def:restricted-k-intersecting-dense

For $C\subseteq V:\abs{C}\geq 1$ and $k\in \Set{0,\dots, \abs{C}-1}$,

$$
\begin{align}
\mc{U}^C_k(x) &:= \arg \overbrace{\max_{\substack{B\subseteq C:\\ \abs{B}\geq k}} f[B](x)}^{f^C_k(x):=}
\end{align}
$$ (restricted-k-intersecting-dense)

where $f[B](x)$ is defined in {eq}`k-intersecting-dense`.

---

+++

---

**Theorem**

For $C\subseteq V:\abs{C}\geq 1$ and $k\in \Set{0,\dots, \abs{C}-1}$,

$$
\sigma_k(B) > \sigma_k(C)\qquad \forall B\in \lim_{x\downarrow \sigma_k(C)} \mc{U}_{k}^C(x),
$$ (larger-strength)

which are solutions associated with the second line segment of $f_k^C(x)$.

---

+++

This follows from the lemma below:

+++

---

**Lemma**

For $C\subseteq V:\abs{C}\geq 1$ and $k\in \Set{0,\dots, \abs{C}-1}$,

$$
\begin{align}
\sigma_k(C)>x \iff f[C](x)> \max_{\substack{B\subseteq C:\nu(C|B)>0\\ \abs{B}\geq k}} f[C](x),
\end{align}
$$ (strength>x)

i.e., $\sigma_k(C)$ in {eq}`strength` is the $x$-coordinate of the first turning point of $f_k^C$. The optimizations in {eq}`strength` and {eq}`strength>x` also share the same set of solutions.

Furthermore,

$$
\begin{align}
\sigma_k(C)=x \implies f[C](x)= \max_{\substack{B\subseteq C:\nu(C|B)>0\\ \abs{B}\geq k}} f[C](x),
\end{align}
$$ (strength=x)

and the converse holds for $k\geq 1$.

---

+++

---

**Theorem**

$$
\begin{align}
C_1\cap C_2 &= \emptyset \text{ or }C_1=C_2 && \forall x\in \mathbb{R}, C_1, C_2\in \mc{C}_{\sigma_1}(x)\\
C_1\cap C_2 &= \emptyset \text{ or }C_1\supseteq C_2 && \forall x_1<x_2, C_1\in \mc{C}_{\sigma_1}(x_1), C_2\in \mc{C}_{\sigma_1}(x_2)
\end{align}
$$ (C1:maximal:hierarchy)

---

+++

This follows from the lemma below.

+++

---

**Lemma**

$$
\sigma_1(C_1\cup C_2)\geq \min \Set{\sigma_1(C_1),\sigma_1(C_2)}\qquad
\forall
C_1, C_2\subseteq V:\abs{C_1\cap C_2}\geq 1
$$ (C1:hierarchy)

---

+++

---

**Proof**

---

+++

---

**Conjecture (NOT TRUE)**

$$
\begin{align}
\abs{C_1\cap C_2} &<k  \text{ or }C_1=C_2 && \forall x\in \mathbb{R}, C_1, C_2\in \mc{C}_{\sigma_k}(x)\\
\abs{C_1\cap C_2} &<k  \text{ or }C_1\supseteq C_2 && \forall x_1<x_2, C_1\in \mc{C}_{\sigma_k}(x_1), C_2\in \mc{C}_{\sigma_k}(x_2)
\end{align}
$$ (Ck:maximal:hierarchy)

or the stronger statement that

$$
\forall k \geq 1, 
\sigma_k(C_1\cup C_2)\geq \min \Set{\sigma_k(C_1),\sigma_k(C_2)}\qquad
\forall
C_1, C_2\subseteq V:\abs{C_1\cap C_2}\geq k.
$$ (Ck:hierarchy)

---

+++

To show {eq}`Ck:hierarchy` implies {eq}`Ck:maximal:hierarchy`, ...

+++

- proof
- graphical networks
- counter-example (minimal, structured) : exists for the stronger statement

Counterexample exists.

When $k=1$

When $k \geq 2$

Suppose $\abs{C_1 \cap C_2} \geq k$

When $\abs{B \cap C_1} \geq k$ or $\abs{B \cap C_2} \geq k$

When $\abs{B \cap C_1} < k$ and $\abs{B \cap C_2} < k$

(in this case and when $B \setminus C_1 \neq \emptyset, B \setminus C_2 \neq \emptyset,,C_1 \neq C_2$, we have $\abs{C_1 \cap C_2 \cap B} \leq k-2$)

```{code-cell} ipython3

```
