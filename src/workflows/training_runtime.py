from typing import Any, Dict, Tuple

from src.training.services.reporting import BatchMetricsRecorder
from src.utils.training_helpers import loader_size
from src.workflows.training_support import select_calibration_source


def execute_training_session_logic(
    session: Any,
    batch_recorder: BatchMetricsRecorder,
    adapter: Any,
    loaders: Dict[str, Any],
    loader_sizes: Dict[str, int],
) -> Tuple[Dict[str, Any], bool, str, Any, Dict[str, Any], Any]:
    try:
        history = session.run()
    finally:
        batch_recorder.flush()
    history_payload = history.to_dict()
    if int(history_payload.get("optimizer_steps", 0)) <= 0:
        raise RuntimeError(
            "Training produced zero optimizer steps. Check the continual split size, batch_size, and "
            "grad_accumulation_steps before launching a full experiment."
        )
    restore_best_state = getattr(session, "restore_best_model_state", None)
    best_state_restored = bool(restore_best_state()) if callable(restore_best_state) else False
    calibration_split_name, calibration_loader = select_calibration_source(loaders, loader_sizes)
    ood_calibration: Dict[str, Any] = {}
    if loader_size(calibration_loader) > 0:
        ood_calibration = adapter.calibrate_ood(calibration_loader)
    return (
        history_payload,
        best_state_restored,
        calibration_split_name,
        calibration_loader,
        ood_calibration,
        getattr(session, "trainer", None),
    )


def run_classifier_rebalance_logic(
    adapter: Any,
    trainer: Any,
    loaders: Dict[str, Any],
    training_cfg: Dict[str, Any],
    colab_cfg: Dict[str, Any],
    telemetry: Any,
    run_id: str,
) -> Tuple[Any, Dict[str, Any], str, Any, Dict[str, Any], bool]:
    rebalance_cfg = dict(training_cfg.get("classifier_rebalance", {}))
    if trainer is None or not bool(rebalance_cfg.get("enabled", False)):
        return trainer, {}, "", None, {}, False

    train_loader = loaders.get("train")
    if train_loader is None or loader_size(train_loader) <= 0:
        return trainer, {}, "", None, {}, False

    log_priors = None
    try:
        log_priors = getattr(trainer, "class_balance_runtime", {})
    except Exception:
        log_priors = None
    # The caller is expected to configure the trainer as needed
    if hasattr(trainer, "configure_classifier_rebalance_stage"):
        trainer.configure_classifier_rebalance_stage(log_priors=log_priors)
    if hasattr(trainer, "set_stage_optimizer_override"):
        trainer.set_stage_optimizer_override(
            learning_rate=float(rebalance_cfg.get("learning_rate", 5e-5)),
            weight_decay=float(rebalance_cfg.get("weight_decay", 0.0)),
        )
    if hasattr(trainer, "setup_stage_optimizer"):
        trainer.setup_stage_optimizer()

    # run rebalance session
    rebalance_session = adapter.build_training_session(
        train_loader=train_loader,
        num_epochs=int(rebalance_cfg.get("epochs", 3)),
        val_loader=loaders.get("val"),
        run_id=f"{run_id}__classifier_rebalance",
        validation_every_n_epochs=int(colab_cfg.get("validation_every_n_epochs", 1)),
    )
    rebalance_history = rebalance_session.run()
    restore_best_state = getattr(rebalance_session, "restore_best_model_state", None)
    rebalance_best_state_restored = bool(restore_best_state()) if callable(restore_best_state) else False
    rebalance_loader_sizes = {str(name): loader_size(loader) for name, loader in loaders.items()}
    calibration_split_name, calibration_loader = select_calibration_source(loaders, rebalance_loader_sizes)
    rebalance_calibration = (
        adapter.calibrate_ood(calibration_loader)
        if calibration_loader is not None and loader_size(calibration_loader) > 0
        else {}
    )
    return (
        getattr(rebalance_session, "trainer", trainer),
        dict(rebalance_history.to_dict()),
        calibration_split_name,
        calibration_loader,
        dict(rebalance_calibration),
        rebalance_best_state_restored,
    )
