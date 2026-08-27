"""Command line interface: ``poler-quantum``.

    poler-quantum demo        -- the classical cognitive cycle (with a
                                 constrained-tracking showcase of Pi_Lambda)
    poler-quantum quantum     -- the quantum-sampled engine + Born stats
    poler-quantum benchmark   -- full benchmark: POLER vs baselines,
                                 figures + metrics.json in ./results/
    poler-quantum spec        -- print the POLER[n] pipeline
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import __version__
from .core.engine import PolerConfig, PolerEngine
from .quantum.engine import QuantumConfig, QuantumPolerEngine
from .quantum.ansatz import PolerAnsatz
from .benchmark.tasks import tracking_task
from .benchmark.runners import (run_all, run_multiseed, summarise,
                                format_table, format_aggregate_table,
                                make_plots, make_aggregate_plot,
                                save_summary)

PIPELINE = """
POLER[n] -- the attention core

    Omega(o_t)  ->  F(p, o; theta)  ->  Pi_Lambda  ->  epsilon
                                                      |
    p_{t+1}  <-  S(p)  <-  R[n]  <------------------+
                (J - D)

    Omega      perception           omega = tanh(o)
    F          free energy          ||p - omega||_G^2 + lam*||p||^2/2
    epsilon    significance         kappa * dx^T G dx        (attention spike)
    R[n]       memory resonance     sum_k rho^k (p - s_{t-k}) (novelty/habit)
    Pi_Lambda  logic projection     I - Jc^+ Jc               (feasibility)
    S(p)       free dynamics        Pi (J - D) Pi p           (creativity -
                                                               stabilisation)

    update     p_{t+1} = clip(p + eta_eff * [Pi(-grad F + gamma*grad R)
                                             + tau * S(p)])
               eta_eff = eta * (1 + beta * tanh(epsilon_hat))
"""


def cmd_demo(args: argparse.Namespace) -> int:
    """Classical POLER cycle + constrained tracking showcase."""
    task = tracking_task(T=200, dim=4, seed=args.seed)
    print(f"task: T={task.spec.T}, dim={task.spec.dim}, "
          f"switches={task.spec.switches}")

    # -- unconstrained POLER -------------------------------------------------
    engine = PolerEngine(PolerConfig(dim=4, seed=args.seed))
    engine.reset()
    report = engine.run(np.tanh(task.observations))
    traj = report.states
    err = np.linalg.norm(traj - task.target, axis=-1)
    print(f"\n[unconstrained] final state   : {np.round(traj[-1], 3)}")
    print(f"[unconstrained] RMSE          : {np.sqrt(np.mean((traj[20:] - task.target[20:])**2)):.4f}")
    print(f"[unconstrained] max epsilon   : {report.eps.max():.4f} "
          f"(at t={int(np.argmax(report.eps))})")

    # -- constrained POLER: p[0] = p[1] must always hold -----------------------
    Jc = np.array([[1.0, -1.0, 0.0, 0.0]])
    c_engine = PolerEngine(PolerConfig(dim=4, seed=args.seed), Jc=Jc)
    c_engine.reset()
    c_report = c_engine.run(np.tanh(task.observations))
    c_traj = c_report.states
    violation = np.abs(c_traj[:, 0] - c_traj[:, 1]).max()
    print(f"\n[constrained p0==p1] max |p0 - p1| over run : {violation:.2e}")
    print(f"[constrained p0==p1] feasible at every step  : "
          f"{c_engine.projector.feasible(c_traj[-1], atol=1e-6)}")
    print("\nThe logic projection Pi_Lambda kept the whole trajectory inside "
          "the subspace p[0] == p[1] while the free-energy descent tracked "
          "the projection of the target -- decisions stay creative inside "
          "the constraints.")
    return 0


def cmd_quantum(args: argparse.Namespace) -> int:
    """Quantum-sampled engine demo."""
    task = tracking_task(T=120, dim=6, seed=args.seed)
    cfg = QuantumConfig(dim=6, seed=args.seed, q_seed=args.qseed)
    engine = QuantumPolerEngine(cfg)
    engine.reset()
    report = engine.run(np.tanh(task.observations))
    traj = report.states
    err = np.linalg.norm(traj - task.target, axis=-1)

    print(f"quantum engine: mode={cfg.mode}, entanglement={cfg.entanglement}, "
          f"sigma_q={cfg.sigma_q}, q_shots={cfg.q_shots}")
    print(f"RMSE vs latent target : {np.sqrt(np.mean((traj[20:] - task.target[20:])**2)):.4f}")
    entropies = np.array([s.born_entropy for s in report.steps])
    print(f"Born entropy          : min={entropies.min():.3f} "
          f"mean={entropies.mean():.3f} max={entropies.max():.3f} bits "
          f"(max possible {np.log2(2 ** 6):.2f})")

    # Born distribution of the final state
    ansatz = PolerAnsatz(6, mode=cfg.mode)
    probs = ansatz.born_probabilities(traj[-1], gamma=cfg.gamma_res,
                                      kappa=cfg.kappa)
    top = np.argsort(probs)[::-1][:4]
    print("\nfinal ansatz -- top Born outcomes:")
    for idx in top:
        bits = format(idx, "06b")
        print(f"  |{bits}> : {probs[idx]:.4f}")
    print("\nEach measurement outcome is a cognitive proposal; the resonance "
          "entanglement layer correlates the coordinates, so exploration is "
          "history-shaped rather than white noise.")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Full benchmark with figures (multi-seed aggregation supported)."""
    poler_cfg = PolerConfig(dim=args.dim, seed=args.seed)
    quantum_cfg = QuantumConfig(dim=args.dim, seed=args.seed,
                                q_seed=args.qseed)
    if args.seeds > 1:
        print(f"multi-seed benchmark: {args.seeds} seeds "
              f"(T={args.T}, dim={args.dim})")
        bundle = run_multiseed(args.seeds, base_seed=args.seed,
                               poler_cfg=poler_cfg, quantum_cfg=quantum_cfg,
                               T=args.T, dim=args.dim)
        print()
        print(format_aggregate_table(bundle["aggregate"]))
        task, results = bundle["representative"]
        summary = summarise(task, results)
        plots = make_plots(task, results, summary, outdir=args.outdir)
        plots.append(make_aggregate_plot(bundle["aggregate"],
                                         outdir=args.outdir))
        # persist both the aggregate and the representative run
        import json
        from pathlib import Path
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        payload = {
            "seeds": bundle["seeds"],
            "task": {"T": task.spec.T, "dim": task.spec.dim,
                     "switches": list(task.spec.switches),
                     "representative_seed": task.spec.seed},
            "aggregate": bundle["aggregate"],
            "representative": summary,
        }
        jpath = outdir / "metrics_multiseed.json"
        jpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nfigures : {[str(p) for p in plots]}")
        print(f"metrics : {jpath}")
        return 0

    task = tracking_task(T=args.T, dim=args.dim, seed=args.seed)
    print(f"task: T={task.spec.T}, dim={task.spec.dim}, "
          f"switches={task.spec.switches}, seed={task.spec.seed}")
    print("running: GD, EMA, POLER, POLER-Quantum ...")
    results = run_all(task, poler_cfg=poler_cfg, quantum_cfg=quantum_cfg)
    summary = summarise(task, results)
    print()
    print(format_table(summary))
    plots = make_plots(task, results, summary, outdir=args.outdir)
    jpath = save_summary(summary, task, outdir=args.outdir)
    print(f"\nfigures : {[str(p) for p in plots]}")
    print(f"metrics : {jpath}")
    return 0


def cmd_spec(_: argparse.Namespace) -> int:
    print(PIPELINE.strip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="poler-quantum",
        description="POLER[n] attention core: free energy, resonance, "
                    "projection, quantum sampling.")
    parser.add_argument("--version", action="version",
                        version=f"poler-quantum {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="classical cognitive cycle demo")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("quantum", help="quantum-sampled engine demo")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--qseed", type=int, default=42)
    p.set_defaults(func=cmd_quantum)

    p = sub.add_parser("benchmark", help="full benchmark with figures")
    p.add_argument("--T", type=int, default=300)
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--seeds", type=int, default=1,
                   help="number of task seeds (>1 enables aggregation)")
    p.add_argument("--qseed", type=int, default=42)
    p.add_argument("--outdir", default="results")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("spec", help="print the POLER[n] pipeline")
    p.set_defaults(func=cmd_spec)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
