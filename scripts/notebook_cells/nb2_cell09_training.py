# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 9.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 6: Training"):
    from scripts.colab_notebook_helpers import (
        build_history_snapshot,
        persist_training_curve_figure,
        persist_training_history_artifacts,
        save_notebook_checkpoint,
    )

    if STATE.get("adapter") is None or STATE.get("loaders") is None:
        raise RuntimeError("Once engine init hucresini calistirin.")

    adapter = STATE["adapter"]
    loaders = STATE["loaders"]
    checkpoint_manager = STATE["checkpoint_manager"]
    effective_params = dict(STATE.get("effective_params") or {})
    val_loader = loaders.get("val")
    run_id = RUN_ID
    telemetry = STATE.get("telemetry") or TELEMETRY

    resume_state = None
    resume_mode = str(effective_params.get("RESUME_MODE", "fresh"))
    resume_manifest = STATE.get("resume_manifest")
    if resume_mode == "resume" and not isinstance(resume_manifest, dict):
        try:
            resume_manifest = checkpoint_manager.get_latest()
            STATE["resume_manifest"] = resume_manifest
        except Exception:
            resume_manifest = None
    if resume_mode == "resume" and isinstance(resume_manifest, dict):
        checkpoint_path = str(resume_manifest.get("path", "")).strip()
        if checkpoint_path:
            try:
                resume_state = adapter.load_training_checkpoint(checkpoint_path)
                STATE["resume_state"] = resume_state
                progress_state = resume_state.get("progress_state") or {}
                print(
                    f"[RESUME] epoch={progress_state.get('epoch', 0)} "
                    f"step={progress_state.get('global_step', 0)}"
                )
            except Exception as exc:
                print(f"[RESUME] Basarisiz: {exc}")

    existing_history = (resume_state or {}).get("history", (resume_state or {}).get("history_snapshot", {}))
    train_loss_curve = list(existing_history.get("train_loss", []))
    val_loss_curve = list(existing_history.get("val_loss", []))
    val_acc_curve = list(existing_history.get("val_accuracy", []))
    macro_f1_curve = list(existing_history.get("macro_f1", []))
    weighted_f1_curve = list(existing_history.get("weighted_f1", []))
    balanced_acc_curve = list(existing_history.get("balanced_accuracy", []))
    gap_curve = list(existing_history.get("generalization_gap", []))

    start_time = time.time()
    session = None
    last_checkpoint_step = -1
    best_val_loss = float(STATE["best_val_loss"]) if STATE.get("best_val_loss") is not None else None

    print(f"[TRAIN] epochs={effective_params['EPOCHS']} device={DEVICE} batch_interval={effective_params['STDOUT_BATCH_INTERVAL']}")
    telemetry.update_latest(
        {
            "phase": "training_started",
            "epochs": int(effective_params["EPOCHS"]),
            "batch_interval": int(effective_params["STDOUT_BATCH_INTERVAL"]),
        }
    )

    def _history_snapshot():
        return build_history_snapshot(
            state_history=STATE.get("history"),
            train_loss_curve=train_loss_curve,
            val_loss_curve=val_loss_curve,
            val_acc_curve=val_acc_curve,
            macro_f1_curve=macro_f1_curve,
            weighted_f1_curve=weighted_f1_curve,
            balanced_acc_curve=balanced_acc_curve,
            gap_curve=gap_curve,
        )

    def _persist_history():
        snapshot = _history_snapshot()
        STATE["history"] = dict((STATE.get("history") or {}), **snapshot)
        persist_training_history_artifacts(
            root=ROOT,
            history_snapshot=STATE["history"],
            telemetry=telemetry,
        )
        return snapshot

    def _checkpoint(reason, event, mark_best=False, val_loss=None):
        if session is None:
            return None
        record = save_notebook_checkpoint(
            checkpoint_manager=checkpoint_manager,
            adapter=adapter,
            session=session,
            reason=reason,
            run_id=run_id,
            telemetry=telemetry,
            mark_best=bool(mark_best),
            val_loss=(float(val_loss) if val_loss is not None else None),
        )
        if record is not None:
            STATE["resume_manifest"] = record
            print(f"[CKPT] {reason} epoch={record.get('epoch', '?')} step={record.get('global_step', '?')}")
        return record

    def session_observer(record):
        global last_checkpoint_step, best_val_loss
        # Ensure matplotlib is lazily loaded before using plt
        _ensure_matplotlib()
        event_type = record.get("event_type", "")
        event = record.get("payload", {})

        if event_type == "stop_requested":
            return

        if event_type == "batch_end":
            batch_num = int(event.get("batch", 0))
            if batch_num > 0 and (batch_num % int(effective_params["STDOUT_BATCH_INTERVAL"]) == 0):
                print(
                    f"[TRAIN] epoch={event.get('epoch', 0)} "
                    f"batch={batch_num}/{event.get('total_batches', 0)} "
                    f"loss={event.get('batch_loss', 0.0):.4f} "
                    f"lr={event.get('lr', 0.0):.6f} "
                    f"speed={event.get('samples_per_sec', 0.0):.1f}s/s "
                    f"elapsed={event.get('elapsed_sec', 0.0):.0f}s eta={event.get('eta_sec', 0.0):.0f}s"
                )
            step = int(event.get("global_step", 0))
            if (
                int(effective_params["CHECKPOINT_EVERY_N_STEPS"]) > 0
                and step > 0
                and (step % int(effective_params["CHECKPOINT_EVERY_N_STEPS"]) == 0)
                and step != last_checkpoint_step
            ):
                _checkpoint(f"batch_{int(effective_params['CHECKPOINT_EVERY_N_STEPS'])}", event)
                last_checkpoint_step = step

        if event_type == "epoch_end":
            train_loss_curve.append(float(event.get("epoch_loss", 0.0)))
            for key, curve in [
                ("val_loss", val_loss_curve),
                ("val_accuracy", val_acc_curve),
                ("macro_f1", macro_f1_curve),
                ("weighted_f1", weighted_f1_curve),
                ("balanced_accuracy", balanced_acc_curve),
                ("generalization_gap", gap_curve),
            ]:
                if key in event:
                    curve.append(float(event[key]))

            val_loss = float(event["val_loss"]) if "val_loss" in event else None
            mark_best = False
            if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
                best_val_loss = val_loss
                STATE["best_val_loss"] = best_val_loss
                mark_best = True

            should_persist_curve = (
                mark_best
                or int(event["epoch_done"]) == 1
                or int(event["epoch_done"]) == int(effective_params["EPOCHS"])
                or bool(event.get("stopped_early", False))
                or (int(event["epoch_done"]) % 5 == 0)
            )
            if should_persist_curve:
                plt.figure(figsize=(13, 3))
                plt.subplot(1, 3, 1)
                plt.plot(range(1, len(train_loss_curve) + 1), train_loss_curve, marker="o", label="Train")
                if val_loss_curve:
                    plt.plot(range(1, len(val_loss_curve) + 1), val_loss_curve, marker="s", label="Val")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Loss")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(1, 3, 2)
                for values, label, marker in [
                    (val_acc_curve, "Acc", "^"),
                    (macro_f1_curve, "MacroF1", "d"),
                    (weighted_f1_curve, "WtdF1", "x"),
                    (balanced_acc_curve, "BalAcc", "*"),
                ]:
                    if values:
                        plt.plot(range(1, len(values) + 1), values, marker=marker, label=label)
                plt.ylim(0, 1)
                plt.xlabel("Epoch")
                plt.ylabel("Score")
                plt.title("Metrics")
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(1, 3, 3)
                if gap_curve:
                    plt.plot(range(1, len(gap_curve) + 1), gap_curve, marker="o", label="Gap")
                plt.axhline(0, color="black", lw=1, alpha=0.5)
                plt.xlabel("Epoch")
                plt.ylabel("Gap")
                plt.title("Gen. Gap")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                persist_training_curve_figure(
                    root=ROOT,
                    epoch_done=int(event["epoch_done"]),
                    telemetry=telemetry,
                )
                plt.close("all")

            _persist_history()

            if mark_best or bool(event.get("stopped_early", False)) or int(event["epoch_done"]) == int(effective_params["EPOCHS"]) or int(effective_params["CHECKPOINT_EVERY_N_STEPS"]) <= 0:
                _checkpoint("epoch_end", event, mark_best=mark_best, val_loss=val_loss)

            parts = [f"[EPOCH] {event['epoch_done']}/{effective_params['EPOCHS']}: train_loss={event.get('epoch_loss', 0.0):.4f}"]
            if "val_loss" in event:
                parts.append(f"val_loss={event['val_loss']:.4f}")
            if "val_accuracy" in event:
                parts.append(f"val_acc={event['val_accuracy']:.4f}")
            if "macro_f1" in event:
                parts.append(f"macro_f1={event['macro_f1']:.4f}")
            if mark_best:
                parts.append("* BEST")
            print(" ".join(parts))
            telemetry.update_latest(
                {
                    "phase": "training",
                    "epoch_done": int(event["epoch_done"]),
                    "global_step": int(event.get("global_step", 0)),
                    "best_val_loss": best_val_loss,
                }
            )

    session = adapter.build_training_session(
        loaders["train"],
        num_epochs=int(effective_params["EPOCHS"]),
        val_loader=val_loader,
        observers=[session_observer],
        resume_state=resume_state,
        run_id=run_id,
        validation_every_n_epochs=int(effective_params["VALIDATION_EVERY_N_EPOCHS"]),
    )

    try:
        history = session.run()
        adapter.is_trained = True
    except KeyboardInterrupt:
        print("[TRAIN] Stop requested by user. Saving latest checkpoint and progress snapshot...")
        telemetry.emit_log("Training stopped by user. Persisting checkpoint.", phase="train", level="warning")
        try:
            STATE["resume_state"] = session.snapshot_state()
        except Exception:
            pass
        if bool(effective_params["CHECKPOINT_ON_EXCEPTION"]):
            try:
                _checkpoint(
                    "manual_stop",
                    {
                        "epoch": 0,
                        "batch": 0,
                        "global_step": int((STATE.get("history") or {}).get("global_step", 0)),
                        "elapsed_sec": time.time() - start_time,
                    },
                )
            except Exception:
                pass
        raise
    except Exception as exc:
        print(f"[TRAIN] Exception: {exc}")
        telemetry.emit_log(f"Training exception: {exc}", phase="train", level="error")
        if bool(effective_params["CHECKPOINT_ON_EXCEPTION"]):
            try:
                _checkpoint(
                    "exception",
                    {
                        "epoch": 0,
                        "batch": 0,
                        "global_step": int((STATE.get("history") or {}).get("global_step", 0)),
                        "elapsed_sec": time.time() - start_time,
                    },
                )
            except Exception:
                pass
        raise

    elapsed_total = time.time() - start_time
    STATE["resume_state"] = session.snapshot_state()
    if history is not None:
        STATE["history"] = history.to_dict()
    elif STATE.get("history") is None:
        STATE["history"] = {}
    _persist_history()
    telemetry.update_latest(
        {
            "phase": "training_complete",
            "elapsed_sec": round(elapsed_total, 3),
            "stopped_early": bool(STATE["history"].get("stopped_early", False)),
            "stopped_by_user": False,
            "continuous_mode": False,
            "completed_cycles": 1,
        }
    )
    print(
        f"[TRAIN] Complete. elapsed={elapsed_total:.1f}s "
        f"stopped_early={STATE['history'].get('stopped_early', False)} "
        f"stopped_by_user=False cycles=1"
    )