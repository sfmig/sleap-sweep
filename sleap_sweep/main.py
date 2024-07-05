import optuna
import sleap

from sleap_sweep.train.config import create_cfg


def objective(trial: optuna.Trial) -> float:
    # define parameters to optimise
    initial_learning_rate_suggest = trial.suggest_float(
        "initial_learning_rate", 1e-5, 1e-2, log=True
    )  # initially: initial_learning_rate= 1e-04

    # create config with selected params
    cfg = create_cfg({"initial_learning_rate": initial_learning_rate_suggest})

    # create a SLEAP Trainer for that config
    trainer = sleap.nn.training.Trainer.from_config(cfg)

    # train model
    trainer.setup()  # is this needed?
    trainer.train()

    # return validation metric to optimise
    val_metrics = sleap.load_metrics(cfg.outputs.run_name, split="val")
    val_metric_optim = 0.5 * (
        val_metrics["vis.precision"] + val_metrics["vis.recall"]
    )

    return val_metric_optim


def main():
    study = optuna.create_study()

    # The optimization finishes after evaluating 1000 times or 3 seconds.
    study.optimize(objective, n_trials=1000, timeout=3)

    print(f"Best params is {study.best_params} with value {study.best_value}")


if __name__ == "__main__":
    main()
