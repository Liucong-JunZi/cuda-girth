# Cross-Level Non-Tree Edge Local Optimality Lemma
# 跨层非树边局部最优性引理

---

## Lemma Statement / 引理陈述

> **Lemma (Local Optimality of Cross-Level Non-Tree Edges).**
> Let $G=(V,E)$ be a simple undirected graph. Perform a standard breadth-first search (BFS) from a source vertex $s$, and let $\ell(v) = 	ext{dist}(s,v)$ denote the BFS level of each vertex $v$. Suppose that while processing vertices at level $t$, the first discovered non-tree edge is a cross-level edge $(u,v)$ with $\ell(u)=t$ and $\ell(v)=t+1$. Then the cycle formed by this edge has length $2t+2$, and:
>
> **引理（跨层非树边局部最优性）。** 设 $G=(V,E)$ 为简单无向图。从源点 $s$ 执行标准广度优先搜索（BFS），令 $\ell(v)=	ext{dist}(s,v)$ 表示各顶点的 BFS 层级。假设在处理第 $t$ 层顶点时，首次发现的非树边为跨层边 $(u,v)$，满足 $\ell(u)=t$，$\ell(v)=t+1$。则该边形成的环长度为 $2t+2$，且：

1. **Intra-level competition / 同层竞争**：
   If there exists another non-tree edge $(x,y)$ within the same level $t$ (i.e., $\ell(x)=\ell(y)=t$), the cycle it forms has length $2t+1$, which is strictly smaller than $2t+2$.

   若第 $t$ 层内存在其他非树边 $(x,y)$（即 $\ell(x)=\ell(y)=t$），其形成的环长度为 $2t+1$，严格小于 $2t+2$。

2. **No shorter cycles in deeper levels / 深层无更短环**：
   Any non-tree edge at level $d \geq t+1$ forms a cycle of length at least $2t+3$, which is strictly larger than $2t+2$.

   任何层级 $d \geq t+1$ 的非树边形成的环长度至少为 $2t+3$，严格大于 $2t+2$。

---

## Corollary (Level Truncation Correctness) / 推论（层级截断正确性）

> **Corollary.** After BFS has finished processing all vertices at level $t$:
> - If an **intra-level** non-tree edge is discovered at this layer, the globally shortest cycle length is $\leq 2t+1$;
> - If only **cross-level** non-tree edges are discovered at this layer (and no intra-level edges exist), the globally shortest cycle length at this stage is $= 2t+2$;
> - **It is unnecessary to continue searching deeper levels** to find a cycle shorter than $2t+2$.
>
> **推论。** 当 BFS 处理完第 $t$ 层的全部顶点后：
> - 若在该层发现**同层**非树边，则全局最短环长度 $\leq 2t+1$；
> - 若该层仅发现**跨层**非树边（且不存在同层非树边），则当前全局最短环长度 $= 2t+2$；
> - **无需继续搜索更深层级**来寻找长度 $< 2t+2$ 的环。

---

## Proof Sketch / 证明概要

Consider any non-tree edge $(x,y)$ at level $d \geq t+1$. By the standard BFS property, non-tree edges can only connect vertices whose levels differ by at most 1. Thus we have two cases:

考虑任意层级 $d \geq t+1$ 的非树边 $(x,y)$。根据标准 BFS 性质，非树边只能连接层级差不超过 1 的顶点。因此有两种情况：

| Case / 情况 | Level relation / 层级关系 | Cycle length / 环长度 | Comparison / 比较 |
|:---|:---|:---|:---|
| Intra-level / 同层 | $\ell(x)=\ell(y)=d$ | $2d+1$ | $\geq 2(t+1)+1 = 2t+3 > 2t+2$ |
| Cross-level / 跨层 | $\vert \ell(x)-\ell(y) \vert =1$ | $2d+2$ |$\geq 2(t+1)+2 = 2t+4 > 2t+2$ |

In both cases, the cycle length is strictly greater than $2t+2$. Therefore, a cycle shorter than $2t+2$ can only be hidden among other non-tree edges **within the same level $t$**, and never appears in level $t+1$ or deeper. $\square$

两种情况下的环长度均严格大于 $2t+2$。因此，长度小于 $2t+2$ 的环只可能隐藏在**同一层 $t$** 的其他非树边中，绝不可能在第 $t+1$ 层或更深出现。$\square$

---

## Engineering Implication / 工程意义

This lemma underpins the correctness of **level-by-level early termination** in GPU-based exact girth computation.

该引理支撑了基于 GPU 的精确围长计算中**层级 BFS 提前终止**的正确性。

> **Key insight / 核心洞察：**
> When the first cross-level non-tree edge is encountered at level $t$, there is no need to "panic" and continue searching deeper levels. Deeper levels can only produce **longer** cycles. The only candidate that could beat the current cross-level edge is another **intra-level** non-tree edge within the same unfinished level $t$. Therefore, the strategy of "finish scanning the current level before stopping" is both **sound and complete**.
>
> 当在第 $t$ 层首次遇到跨层非树边时，无需"恐慌"地继续往深层搜索。更深层级只可能产生**更长**的环。唯一可能击败当前跨层边的候选者，是同一尚未完成的第 $t$ 层内的另一条**同层**非树边。因此，"把当前层扫完再停"的策略是**既正确又完备**的。

---

## Notation Reference / 符号说明

| Symbol / 符号 | Meaning / 含义 |
|:---|:---|
| $G=(V,E)$ | Simple undirected graph / 简单无向图 |
| $s$ | BFS source vertex / BFS 源点 |
| $\ell(v)$ | BFS level (distance from $s$) / BFS 层级（到 $s$ 的距离） |
| $(u,v)$ | Non-tree edge discovered during BFS / BFS 过程中发现的非树边 |
| $t$ | The level where the first non-tree edge appears / 首次出现非树边的层级 |
| $2t+1$ | Length of intra-level cycle / 同层非树边形成的环长度（奇数） |
| $2t+2$ | Length of cross-level cycle / 跨层非树边形成的环长度（偶数） |

---

*Document prepared for the cuda-girth project / 为 cuda-girth 项目准备的理论文档。*
