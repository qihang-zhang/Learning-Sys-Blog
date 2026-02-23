---
categories: ["Reinforcement Learning", "Product of Experts"]
date: 2025-10-20
draft: false
comments: true
links:
readtime: 8
slug: ppo-wpoe-cheatsheet
authors:
  - <qihang>
---

# PPO, gPoE, and Naive Bayes — Quick Notes

Compact derivations connecting Naive Bayes with Laplace smoothing, weighted/product-of-experts intuition, and PPO-style RL objectives.

## Naive Bayes with Laplace Smoothing

### Naive Bayes Formulation

$$
\begin{align}
    P_N(x_i \mid x_{<i}) &\quad \text{(conditional probability)} \\
    P_N(x_{\leq t}) &= \prod_{i=1}^{t} P_N(x_i \mid x_{<i})
\end{align}
$$

### Target Distribution via Product of Experts

$$
\begin{align}
    \pi^*_\theta(x_i \mid x_{<i}) = \frac{P_N(x_i \mid x_{<i})^\alpha \cdot P_\theta(x_i \mid x_{<i})^{1-\alpha}}{Z}
\end{align}
$$

### Corresponding Optimization Objective

$$
\begin{align}
    &\text{maximize } \mathbb{E}_{\pi_\theta(x_i \mid x_{<i})} \left[ \alpha \cdot \log P_N(x_i \mid x_{<i}) + (1 - \alpha) \log P_\theta(x_i \mid x_{<i}) \right] + \mathbb{E}_{\pi_\theta(x_i \mid x_{<i})} \log \frac{1}{\pi_\theta(x_i \mid x_{<i})} \notag \\
    \Leftrightarrow&\text{maximize } \mathbb{E}_{\pi_\theta(x_i \mid x_{<i})} \left[ \log P_N(x_i \mid x_{<i}) - \log P_\theta(x_i \mid x_{<i}) \right] + \frac{1}{\alpha}\mathbb{E}_{\pi_\theta(x_i \mid x_{<i})} \log \frac{P_\theta(x_i \mid x_{<i})}{\pi_\theta(x_i \mid x_{<i})} \notag \\
    \Leftrightarrow&\text{maximize } \mathbb{E}_{\pi_\theta(x_i \mid x_{<i})} \left[ \log P_N(x_i \mid x_{<i}) - \log P_\theta(x_i \mid x_{<i}) \right] - \frac{1}{\alpha} D_{\text{KL}}\Bigl(\pi_\theta(\cdot \mid x_{<i}) \| P_\theta(\cdot \mid x_{<i}) \Bigr)
\end{align}
$$

### Reward Definition Under PPO

$$
\begin{align}
    r_{\text{RM}} = \log P_N(x_i \mid x_{<i}) - \log P_\theta(x_i \mid x_{<i})
\end{align}
$$

## Adjust PPO Target According to the Experience of wPoE

Define the target distribution:

$$
\begin{align}
    \pi^*_\theta(y \mid x) = \frac{\exp\left( \alpha r_{\text{RM}}(x, y) + (1 - \alpha) \log \pi_{\text{ref}}(y \mid x) \right)}{Z}
\end{align}
$$

This corresponds to:

$$
\begin{align}
    &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \alpha r_{\text{RM}}(x, y) + (1 - \alpha) \log \pi_{\text{ref}}(y \mid x) \right] + H(\pi_\theta) \\
    \Leftrightarrow &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \alpha r_{\text{RM}}(x, y) + (1 - \alpha) \log \pi_{\text{ref}}(y \mid x) \right] + \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \alpha r_{\text{RM}}(x, y) + (-\alpha) \log \pi_{\text{ref}}(y \mid x) \right] + \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \log \frac{\pi_{\text{ref}}(y \mid x)}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{maximize } \mathbb{E}_{y \sim \pi_\theta} \left[ \alpha \left[ r_{\text{RM}}(x, y) - \log \pi_{\text{ref}}(y \mid x) \right] \right] - D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\end{align}
$$

Final form:

$$
\begin{align}
    \Leftrightarrow \text{maximize } \mathbb{E}_{y \sim \pi_\theta} \left[ r_{\text{RM}}(x, y) - \log \pi_{\text{ref}}(y \mid x) \right] - \frac{1}{\alpha} D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
\end{align}
$$

> **Remark:** Remove the influence of $\pi_{\text{ref}}$ in the reward model to implement a clean test-time version.

## Optimization Goal of PPO

$$
\begin{align}
    &\text{maximize } \mathbb{E}_{x \sim P_{\text{data}}} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r_{\text{RM}}(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]\\
    \Leftarrow &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r_{\text{RM}}(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right], \text{ for all } x \in X
\end{align}
$$

$$
\begin{align}
    \pi^*_\theta(\cdot \mid x) =& \underset{\pi_\theta \in \Pi}{\text{argmax }} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[r_{\text{RM}}(x, y) + \beta \log \frac{\pi_{\text{ref}}(y \mid x)}{\pi_\theta(y \mid x)} \right] \notag \\
    \pi^*_\theta(\cdot \mid x) =& \underset{\pi_\theta \in \Pi}{\text{argmax }} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[\Bigl(r_{\text{RM}}(x, y) + \beta\log \pi_{\text{ref}}(y \mid x)\Bigr) + \beta \log \frac{1}{\pi_\theta(y \mid x)} \right]
\end{align}
$$

## Connect PPO with gPoE via Max Entropy RL

Following the conclusion from [this blog post](https://blog.qihang-zhang.com/2025/10/06/max-ent-rl-and-boltzmann-distribution#optimal-soft-policy-form):

$$
\begin{align}
    \pi^*_\theta(\cdot \mid x) =& \frac{\exp\Bigl( (r_{\text{RM}}(x, y))/\beta + \log \pi_{\text{ref}}(y \mid x) \Bigr)}{Z} \notag \\
    =& \frac{\exp\Bigl(r_{\text{RM}}(x, y)\Bigr)^{\frac{1}{\beta}} \cdot \pi_{\text{ref}}(y \mid x) }{Z}   \quad \text{gPoE Here!}
\end{align}
$$

$$
\begin{align}
    \log \pi^*_\theta(\cdot \mid x) =& \frac{1}{\beta} r_{\text{RM}}(x, y) + \log \pi_{\text{ref}}(y \mid x) - \log Z \notag \\
    \text{logits } \pi^*_\theta(\cdot \mid x) =& \frac{1}{\beta} r_{\text{RM}}(x, y) + \log \pi_{\text{ref}}(y \mid x)
\end{align}
$$

## From PPO to SFT

### Classical PPO in RL

**PPO Target:**

$$
\begin{align}
    &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r_{\text{RM}}(x, y) + \beta \log \frac{\pi_{\text{ref}}(y \mid x)}{\pi_\theta(y \mid x)} \right] \\
    \Leftrightarrow &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r_{\text{RM}}(x, y) + \beta \log \pi_{\text{ref}}(y \mid x) \right] + \beta \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{maximize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r_{\text{RM}}(x, y) + \beta \log \pi_{\text{ref}}(y \mid x) \right] + \beta H(\pi_\theta(\cdot \mid x)) \\
    \Leftrightarrow \pi^*_\theta(y \mid x) =& \underset{\pi_\theta}{\text{argmax }} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ r_{\text{RM}}(x, y) + \beta \log \pi_{\text{ref}}(y \mid x) \right] + \beta H(\pi_\theta(\cdot \mid x))
\end{align}
$$

Following the optimal policy form from Max Entropy RL:

$$
\begin{align}
    \pi^*_\theta(\cdot \mid x) &= \frac{\exp\left( \frac{r_{\text{RM}}(x, y)}{\beta} + \log \pi_{\text{ref}}(y \mid x) \right)}{Z} \\
    &= \frac{\exp\left( \frac{r_{\text{RM}}(x, y)}{\beta} \right)}{Z} \cdot \pi_{\text{ref}}(y \mid x) \\
    &= \frac{\exp\left( r_{\text{RM}}(x, y) / \beta \right)}{Z(r_{\text{RM}}, \pi_{\text{ref}})} \cdot \pi_{\text{ref}}(y \mid x)
\end{align}
$$

Define $\pi_{\text{RM}}$ as:

$$
\begin{align}
    \pi_{\text{RM}} = \frac{\exp(r_{\text{RM}}(x, y) / \beta)}{Z(r_{\text{RM}})}
\end{align}
$$

This gives us: $\pi_{\text{RM}} \otimes \pi_{\text{ref}}$ (Product of Experts).

### SFT Target

Assume $\pi_{\text{ref}}$ corresponds to a distribution $P_{\text{data}}(x, y)$.

**Target:** data set $D \Leftrightarrow$ distribution $P_{\text{data}}(x, y)$

$$
\begin{align}
    &\text{minimize } D_{\text{KL}}(P_{\text{data}}, \pi_\theta) \\
    \Leftrightarrow &\text{minimize } \mathbb{E}_{x, y \sim P_{\text{data}}} \log \frac{P_{\text{data}}(y \mid x)}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{minimize } \mathbb{E}_{x, y \sim P_{\text{data}}} \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{minimize } \mathbb{E}_{x \sim P_{\text{data}}} \mathbb{E}_{y \sim P_{\text{data}}(\cdot \mid x)} \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftarrow &\text{minimize } \mathbb{E}_{y \sim P_{\text{data}}(\cdot \mid x)} \log \frac{1}{\pi_\theta(y \mid x)}, \quad \text{for all } x \in X
\end{align}
$$

$$
\begin{align}
    \Leftrightarrow &\text{minimize } \sum_{y \in Y} P_{\text{data}}(y \mid x) \cdot \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{minimize } \sum_{y \in Y} \pi_\theta(y \mid x) \cdot \frac{P_{\text{data}}(y \mid x)}{\pi_\theta(y \mid x)} \cdot \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{minimize } \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \frac{P_{\text{data}}(y \mid x)}{\pi_\theta(y \mid x)} \cdot \log \frac{1}{\pi_\theta(y \mid x)} \\
    \Leftrightarrow &\text{maximize } \log \pi_\theta(y \mid x) \quad \text{with weight } \frac{P_{\text{data}}(y \mid x)}{\pi_\theta(y \mid x)}
\end{align}
$$
