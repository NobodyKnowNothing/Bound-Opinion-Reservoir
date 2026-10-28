"""Hegselmann-Krause bounded-confidence dynamics on a social graph, driven by an input signal.

Opinions live in [-1, 1]. Each step every agent:

1. looks at the agents it follows (plus itself) whose opinions lie within the
   confidence bound ``eps`` of its own, and moves a fraction ``mu`` of the way
   toward their weighted mean (HK update restricted to the graph). With
   ``outrage = 0`` every visible account counts equally. With ``outrage > 0`` the
   feed amplifies extreme voices: account j's weight is exp(outrage * |x_j|), so
   the most extreme visible opinions dominate what an agent averages over;
2. if it is "informed" (consumes media), receives the input signal ``u(t)``
   scaled by its own susceptibility ``w_in[i]`` and the global ``input_gain``
   (plain input scaling: how loud the news is, not how outrageous);
3. is pulled back toward its own prior opinion at its personal stubbornness rate
   (Friedkin-Johnsen anchoring; this is what gives the reservoir fading memory);
4. gets a little idiosyncratic noise, and is clipped back to [-1, 1].

The opinion vector over time is the reservoir state.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    n_agents: int = 200
    k_follow: int = 10          # out-degree: how many accounts each agent sees
    eps: float = 0.3            # confidence bound
    mu: float = 0.5             # social learning rate
    input_gain: float = 0.3     # input scaling: how hard the news hits informed agents
    outrage: float = 0.0        # attention amplification of extreme opinions in the feed
    informed_frac: float = 0.3  # fraction of agents exposed to the input
    noise: float = 0.01         # idiosyncratic opinion noise
    anchor: tuple = (0.02, 0.2) # range of per-agent stubbornness (pull back toward prior opinion)
    seed: int = 0


def random_follow_graph(n, k, rng):
    """Directed graph: A[i, j] = 1 if i sees j. Every agent follows exactly k others."""
    A = np.zeros((n, n), dtype=bool)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        A[i, rng.choice(others, size=k, replace=False)] = True
    return A


@dataclass
class OpinionReservoir:
    p: Params
    A: np.ndarray = field(init=False)
    w_in: np.ndarray = field(init=False)
    x: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.p.seed)
        self.rng = rng
        n = self.p.n_agents
        self.A = random_follow_graph(n, self.p.k_follow, rng)
        informed = rng.random(n) < self.p.informed_frac
        # Susceptibility: sign says which way the agent reads the news, magnitude how hard.
        self.w_in = informed * rng.uniform(-1, 1, n)
        self.prior = rng.uniform(-1, 1, n)
        self.lam = rng.uniform(*self.p.anchor, n)
        self.x = self.prior.copy()

    def step(self, u):
        p, x = self.p, self.x
        close = np.abs(x[None, :] - x[:, None]) < p.eps
        M = (self.A & close) | np.eye(p.n_agents, dtype=bool)
        if p.outrage:
            # Shifted by -1 so the largest weight is exp(0); the shift cancels in the normalisation.
            M = M * np.exp(p.outrage * (np.abs(x) - 1))[None, :]
        social_mean = (M @ x) / M.sum(1)
        x = (1 - p.mu) * x + p.mu * social_mean
        x = x + self.lam * (self.prior - x)
        x = x + p.input_gain * self.w_in * u + p.noise * self.rng.standard_normal(p.n_agents)
        self.x = np.clip(x, -1.0, 1.0)
        return self.x

    def run(self, u):
        """Drive the reservoir with the signal u (shape T); return states (T, N)."""
        return np.stack([self.step(ut).copy() for ut in u])
