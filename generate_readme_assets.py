from pathlib import Path

import matplotlib
import numpy as np

from bandit_task import (
    BanditEnvironment,
    QLearningModel,
    plot_parameter_recovery,
    simulate_qlearning_agent,
)


matplotlib.use("Agg")

OUTPUT_DIR = Path("assets")
OUTPUT_PATH = OUTPUT_DIR / "parameter-recovery.png"


def generate_parameter_recovery_figure() -> None:
    rng = np.random.default_rng(99)

    true_alphas, true_betas = [], []
    recovered_alphas, recovered_betas = [], []

    n_participants = 40

    for i in range(n_participants):
        true_alpha = rng.uniform(0.05, 0.95)
        true_beta = rng.uniform(0.5, 15.0)

        task_env = BanditEnvironment(4, [0.2, 0.5, 0.8, 0.35], seed=i)
        result = simulate_qlearning_agent(
            task_env,
            alpha=true_alpha,
            beta=true_beta,
            n_trials=150,
            seed=i,
        )

        model = QLearningModel(n_arms=4)
        recovered = model.fit(result["choices"], result["rewards"], n_restarts=5)

        true_alphas.append(true_alpha)
        true_betas.append(true_beta)
        recovered_alphas.append(recovered["alpha"])
        recovered_betas.append(recovered["beta"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig = plot_parameter_recovery(
        true_alphas=true_alphas,
        true_betas=true_betas,
        recovered_alphas=recovered_alphas,
        recovered_betas=recovered_betas,
        show=False,
    )
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    fig.clf()
    print(f"Saved parameter recovery figure to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    generate_parameter_recovery_figure()
