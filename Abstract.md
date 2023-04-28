---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Beyond Maximal Densest Subgraphs: <br>A General Framework for Information Density

*Chung Chan* and *Chao Zhao*  
*Department of Computer Science*  
*City University of Hong Kong*

+++

#### Acknowledgements

This a joint work with Ali Al-Bashabsheh (School of General Engineering, Beijing, China).

+++ {"tags": []}

## Abstract

+++

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/git/https%3A%2F%2Fgitlab1.cs.cityu.edu.hk%2Fccha23%2Fcs5483jupyter/main?urlpath=git-pull?repo%3Dhttps%3A%2F%2Fgitlab1.cs.cityu.edu.hk%2Fccha23%2Fcs5483jupyter%26branch%3Dmain%26urlpath%3Dlab%2Ftree%2Fcs5483jupyter%2Fdocs%2FAbstract.ipynb)
[![CS5483](badge.dio.svg)](https://cs5483.cs.cityu.edu.hk/hub/user-redirect/git-pull?repo=https://gitlab1.cs.cityu.edu.hk/ccha23/cs5483jupyter&urlpath=lab/tree/cs5483jupyter/docs/Abstract.ipynb&branch=main)

+++ {"jp-MarkdownHeadingCollapsed": true, "tags": []}

Finding densest $k$-subgraphs is a well-known NP-hard problem, but it is useful to identify clusters that are tightly connected subsets. Existing work cannot find the densest $k$-subgraphs that are non-maximal. We defined a new notion of strength for the subsets in the network with a higher-order dependency structure modeled by a supermodular function. By using the notion of strength, we proposed a framework for finding maximal strong communities, which enables us to identify more densest $k$-subgraphs in polynomial time than existing work. In particular, many existing works are confined to finding maximal densest $k$-subgraphs. There are also compact dense subgraphs that are not the densest $k$-subgraphs, and our framework for finding maximal strong communities can identify them. Unlike existing work that defines locally densest subgraphs with no natural extensions to weighted graphs, our framework covers weighted/unweighted directed/undirected graphs/hypergraph as special cases. It also applies to a general notion of the density of information well beyond weighted graphs.

```{code-cell} ipython3

```
