# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 12.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 9: Final Evaluation"):
    import json
    from datetime import datetime, timezone
    from scripts.colab_notebook_helpers import (
        merge_training_summary_fields,
        notebook_artifact_root,
        persist_behavioral_acceptance_artifact,
        persist_behavioral_dev_report_artifact,
        persist_production_readiness_artifact,
        persist_training_run_context_artifact,
        persist_validation_artifacts,
        print_notebook_optimization_campaign_status,
        resolve_notebook_optimization_campaign,
        summarize_notebook_optimization_campaign,
    )
    from src.adapter.independent_crop_adapter import IndependentCropAdapter
    from src.training.notebook_runtime_helpers import select_notebook_ood_policy
    from src.training.services.ood_benchmark import run_leave_one_class_out_benchmark
    from src.training.services.ood_score_selection import apply_primary_score_method_to_evaluation
    from src.training.validation import evaluate_model_with_artifact_metrics

    if STATE.get("adapter") is None or STATE.get("loaders") is None:
        raise RuntimeError("Once engine init hucresi calistirilmali.")

    adapter = STATE["adapter"]
    loaders = STATE["loaders"]
    effective_params = dict(STATE.get("effective_params") or {})
    dataset_lineage = dict(STATE.get("dataset_lineage") or {})
    selected_dataset_root = Path(STATE.get("selected_dataset_root") or "")
    if not dataset_lineage:
        raise RuntimeError("Dataset lineage was not resolved; rerun the dataset validation cell.")
    selection_only = bool(globals().get("RECOVERY_SELECTION_ONLY", False))
    test_loader = loaders.get("test")
    if not selection_only and (test_loader is None or len(test_loader.dataset) == 0):
        raise RuntimeError("Test loader bos. Final degerlendirme held-out test split ile yapilmali.")

    trainer = adapter._trainer
    if trainer is None:
        raise RuntimeError("Trainer hazir degil.")

    notebook_config = json.loads(json.dumps(BASE_CONFIG))
    notebook_config.setdefault("training", {})["continual"] = json.loads(json.dumps(STATE["continual_config"]))
    evaluation_cfg = notebook_config.get("training", {}).get("continual", {}).get("evaluation", {})
    artifact_root = notebook_artifact_root(ROOT)
    from src.workflows.training_utils import collect_dataset_manifest_context

    persist_training_run_context_artifact(
        root=ROOT,
        context_payload={
            "schema_version": "v1_training_run_context",
            "run_id": RUN_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "surface": "notebook_2",
            "crop_name": CROP_NAME,
            "part_name": PART_NAME,
            "resolved_config": dict(notebook_config),
            "dataset": {
                "crop_root": str(selected_dataset_root.resolve()),
                "crop_name": CROP_NAME,
                "part_name": PART_NAME,
                "dataset_key": str(STATE.get("runtime_dataset_key") or DATASET_NAME),
                "resolution_source": "notebook_2_validated_runtime_target",
                "manifests": collect_dataset_manifest_context(selected_dataset_root),
                "lineage": dataset_lineage,
            },
        },
        telemetry=TELEMETRY,
    )

    trainer.adapter_model.eval()
    trainer.classifier.eval()
    trainer.fusion.eval()

    classes = [name for name, _ in sorted(adapter.class_to_idx.items(), key=lambda item: item[1])]
    ood_test_loader = loaders.get("ood")
    ood_dev_loader = loaders.get("ood_dev")
    policy = dict(STATE.get("ood_policy_selection") or {})
    if not policy:
        policy = select_notebook_ood_policy(
            trainer=trainer,
            loaders=loaders,
            continual_config=dict(STATE.get("continual_config") or {}),
        )
        STATE["ood_policy_selection"] = policy
    requested_primary_score_method = str(policy.get("requested_primary_score_method", "auto"))
    selected_primary_score_method = str(policy.get("selected_primary_score_method", "ensemble"))
    selection_source = str(policy.get("selection_source", "configured"))

    def _evaluate_split(split_name: str, loader, ood_evidence_loader, *, artifact_subdir: str, label: str):
        evaluation = evaluate_model_with_artifact_metrics(trainer, loader, ood_loader=ood_evidence_loader)
        if evaluation is None:
            raise RuntimeError("Degerlendirme ornegi bulunamadi.")
        evaluation = apply_primary_score_method_to_evaluation(
            evaluation,
            selected_primary_score_method,
            requested_primary_score_method=requested_primary_score_method,
            selection_source=selection_source,
        )
        artifacts = persist_validation_artifacts(
            root=ROOT,
            y_true=evaluation.y_true,
            y_pred=evaluation.y_pred,
            classes=classes,
            telemetry=TELEMETRY,
            artifact_subdir=artifact_subdir,
            require_ood=bool(evaluation_cfg.get("require_ood_for_gate", True)),
            emit_metric_gate=bool(evaluation_cfg.get("emit_ood_gate", True)),
            ood_labels=evaluation.ood_labels,
            ood_scores=evaluation.ood_scores,
            ood_scores_by_method=evaluation.ood_scores_by_method,
            sure_ds_f1=evaluation.sure_ds_f1,
            conformal_empirical_coverage=evaluation.conformal_empirical_coverage,
            conformal_avg_set_size=evaluation.conformal_avg_set_size,
            ood_type_breakdown=evaluation.ood_type_breakdown,
            prediction_rows=evaluation.prediction_rows,
            context={
                "crop_name": CROP_NAME,
                "part_name": PART_NAME,
                "run_id": RUN_ID,
                "split_name": split_name,
                **evaluation.context,
            },
        )
        metrics = artifacts["metric_gate"]["metrics"]
        extras = []
        if metrics.get("ood_auroc") is not None:
            extras.append(f"ood_auroc={float(metrics['ood_auroc']):.4f}")
        if metrics.get("sure_ds_f1") is not None:
            extras.append(f"sure_ds_f1={float(metrics['sure_ds_f1']):.4f}")
        if metrics.get("conformal_empirical_coverage") is not None:
            extras.append(f"conformal_cov={float(metrics['conformal_empirical_coverage']):.4f}")
        suffix = " " + " ".join(extras) if extras else ""
        accuracy = float(artifacts["report_dict"].get("accuracy", 0.0))
        print(f"[{label}] ornek={len(evaluation.y_true)} sinif={len(classes)} accuracy={accuracy:.4f}{suffix}")
        return artifacts

    results = {}
    val_loader = loaders.get("val")
    if selection_only and (val_loader is None or len(val_loader.dataset) == 0):
        raise RuntimeError("Recovery candidate selection requires a non-empty ID validation loader.")
    if selection_only and (ood_dev_loader is None or len(ood_dev_loader.dataset) == 0):
        raise RuntimeError("Recovery candidate selection requires a non-empty OOD-dev loader.")
    if val_loader is not None and len(val_loader.dataset) > 0:
        results["validation"] = _evaluate_split(
            "val",
            val_loader,
            ood_dev_loader,
            artifact_subdir="validation",
            label="DOGRULAMA (referans)",
        )

    if selection_only:
        selected_artifacts = results["validation"]
        dev_report = persist_behavioral_dev_report_artifact(
            root=ROOT,
            classification_metrics=dict(selected_artifacts.get("metric_gate", {}).get("metrics", {})),
            prediction_rows=list(selected_artifacts.get("prediction_rows") or []),
            context={
                "run_id": RUN_ID,
                "crop_name": CROP_NAME,
                "part_name": PART_NAME,
                "ood_scope": "same_crop_same_part_unsupported_disease",
                "router_scope_excluded": ["wrong_part", "other_plant", "non_plant"],
            },
            telemetry=TELEMETRY,
        )
        STATE["evaluation_artifacts"] = results
        STATE["behavioral_dev_report"] = dev_report["payload"]
        STATE["behavioral_acceptance"] = {}
        STATE["production_readiness"] = {}
        _ensure_matplotlib()
        plt.close("all")
        print(
            f"[DEV-SELECTION] authoritative=false status={dev_report['payload'].get('status', 'fail')} "
            f"pass={dev_report['payload'].get('pass', False)} locked_test_loaded=false"
        )
        TELEMETRY.update_latest(
            {
                "phase": "recovery_dev_evaluation_complete",
                "evaluation_splits": ["validation"],
                "adapter_behavioral_dev_report": dev_report["payload"].get("status"),
                "locked_test_evidence_loaded": False,
            }
        )
        print("[DONE] ID validation and OOD-dev selection artifacts saved; locked test remains unopened.")
    else:
        results["test"] = _evaluate_split(
            "test",
            test_loader,
            ood_test_loader,
            artifact_subdir="test",
            label="TEST (ayrilmis)",
        )
        selected_artifacts = results["test"]
        real_ood_present = ood_test_loader is not None and len(ood_test_loader.dataset) > 0
        ood_evidence_source = "real_ood_split" if real_ood_present else "unavailable"
        ood_evidence_metrics = dict(selected_artifacts["metric_gate"]["metrics"]) if real_ood_present else {}
        benchmark_summary = {}
        if (
            not real_ood_present
            and str(evaluation_cfg.get("ood_fallback_strategy", "held_out_benchmark")) == "held_out_benchmark"
            and bool(evaluation_cfg.get("ood_benchmark_auto_run", True))
        ):
            print("[OOD] Gercek OOD split bulunamadi; held-out benchmark fallback calisiyor...")
            benchmark_summary = run_leave_one_class_out_benchmark(
                crop_name=CROP_NAME,
                class_names=classes,
                loaders=loaders,
                config=notebook_config,
                device=DEVICE,
                artifact_root=artifact_root,
                adapter_factory=IndependentCropAdapter,
                run_id=RUN_ID,
                num_epochs=int(effective_params["EPOCHS"]),
                telemetry=TELEMETRY,
                emit_event=lambda event_type, payload: TELEMETRY.emit_event(event_type, payload, phase="evaluation"),
                min_classes=int(evaluation_cfg.get("ood_benchmark_min_classes", 3)),
            )
            ood_evidence_source = "held_out_benchmark"
            ood_evidence_metrics = dict(benchmark_summary.get("metrics", {}))

        readiness = persist_production_readiness_artifact(
            root=ROOT,
            classification_metric_gate=selected_artifacts.get("metric_gate"),
            classification_split="test",
            ood_evidence_source=ood_evidence_source,
            ood_metrics=ood_evidence_metrics,
            context={
                "run_id": RUN_ID,
                "crop_name": CROP_NAME,
                "part_name": PART_NAME,
                "classification_split": "test",
                "ood_benchmark_status": benchmark_summary.get("status"),
                "ood_benchmark_passed": benchmark_summary.get("passed"),
                "ood_requested_primary_score_method": requested_primary_score_method,
                "ood_primary_score_method": selected_primary_score_method,
                "ood_primary_score_selection_source": selection_source,
                "ood_primary_score_selection": dict(policy.get("selection") or {}),
                "dataset_lineage": dataset_lineage,
                "require_immutable_dataset_lineage": dataset_lineage.get("source_kind") != "local_legacy",
            },
            require_ood=bool(evaluation_cfg.get("require_ood_for_gate", True)),
            telemetry=TELEMETRY,
        )
        behavioral = persist_behavioral_acceptance_artifact(
            root=ROOT,
            classification_metrics=dict(selected_artifacts.get("metric_gate", {}).get("metrics", {})),
            prediction_rows=list(selected_artifacts.get("prediction_rows") or []),
            context={
                "run_id": RUN_ID,
                "crop_name": CROP_NAME,
                "part_name": PART_NAME,
                "classification_split": "test",
                "ood_scope": "same_crop_same_part_unsupported_disease",
                "router_scope_excluded": ["wrong_part", "other_plant", "non_plant"],
                "production_readiness_status": readiness["payload"].get("status"),
                "frozen_dev_winner": True,
                "dataset_lineage": dataset_lineage,
            },
            telemetry=TELEMETRY,
        )

        STATE["evaluation_artifacts"] = results
        STATE["ood_benchmark"] = benchmark_summary
        STATE["behavioral_acceptance"] = behavioral["payload"]
        STATE["production_readiness"] = readiness["payload"]
        _ensure_matplotlib()
        plt.close("all")
        print(
            f"[ACCEPTANCE] status={behavioral['payload'].get('status', 'fail')} "
            f"pass={behavioral['payload'].get('pass', False)}"
        )
        print(f"[READINESS-COMPAT] status={readiness['payload'].get('status', 'failed')}")
        TELEMETRY.update_latest(
            {
                "phase": "evaluation_complete",
                "evaluation_splits": sorted(results.keys()),
                "ood_evidence_source": readiness["payload"].get("ood_evidence_source"),
                "adapter_behavioral_acceptance": behavioral["payload"].get("status"),
                "production_readiness": readiness["payload"].get("status"),
            }
        )
        print("[DONE] Frozen winner evaluated once on held-out ID test and OOD test.")

from scripts.colab_notebook_helpers import build_notebook_completion_report, maybe_auto_disconnect_colab_runtime, merge_training_summary_fields, print_notebook_optimization_campaign_status, resolve_notebook_optimization_campaign, summarize_notebook_optimization_campaign

REPO_RUN_EXPORTS = save_run_outputs_to_repo()
effective_params = dict(STATE.get("effective_params") or {})
resolved_dataset_root = Path(STATE.get("runtime_dataset_root") or RUNTIME_DATASET_ROOT)
resolved_dataset_key = str(STATE.get("runtime_dataset_key") or STATE.get("selected_dataset_name") or DATASET_NAME or "").strip()
if resolved_dataset_key:
    optimization_campaign_mode = "continue" if bool(effective_params.get("ENABLE_BAYESIAN_OPTIMIZATION", True)) else "disabled"
    STATE["optimization_campaign"] = resolve_notebook_optimization_campaign(
        root=ROOT,
        runtime_dataset_root=resolved_dataset_root,
        dataset_key=resolved_dataset_key,
        crop_name=CROP_NAME,
        part_name=PART_NAME,
        backbone_model_name=str(dict(BASE_CONFIG or {}).get("training", {}).get("continual", {}).get("backbone", {}).get("model_name", "")),
        notebook_parameters=effective_params,
        dataset_lineage=dataset_lineage,
        mode=optimization_campaign_mode,
        telemetry=TELEMETRY,
    )
    STATE["recommendation_report"] = summarize_notebook_optimization_campaign(STATE["optimization_campaign"])
    STATE["recommendation_decision"] = str(STATE["optimization_campaign"].get("status", "disabled"))
    print_notebook_optimization_campaign_status(STATE["optimization_campaign"], print_fn=print)
notebook_export_result = export_current_colab_notebook(REPO_NOTEBOOK_OUTPUT_PATH)

extra_entries = [
    {
        "path": build_adapter_bundle_root(REPO_OUTPUT_DIR, CROP_NAME, PART_NAME) / "continual_sd_lora_adapter",
        "category": "adapter_export",
        "priority": "high",
        "title_tr": "Repo mirror adapter export klasoru",
        "description_tr": "Repo mirror icindeki adapter export klasoru.",
        "reader_goal": "Export edilen adapter klasorunu bulmak",
        "generated_by": "notebook_2",
        "decision_importance": "deploy_handoff",
        "read_order": 70,
    },
    {
        "path": REPO_NOTEBOOK_OUTPUT_PATH,
        "category": "adapter_export",
        "priority": "medium",
        "title_tr": "Calistirilmis notebook exportu",
        "description_tr": "Bu kosuda calisan notebook'un kaydedilmis kopyasi.",
        "reader_goal": "Notebook'u ayni ciktiyla tekrar incelemek",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 71,
    },
    {
        "path": REPO_TELEMETRY_DIR / "events.jsonl",
        "category": "logs_and_checkpoints",
        "priority": "medium",
        "title_tr": "Telemetry event logu",
        "description_tr": "Notebook olayi bazli telemetry kaydi.",
        "reader_goal": "Notebook akisini olay bazinda incelemek",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 80,
    },
    {
        "path": REPO_TELEMETRY_DIR / "runtime.log",
        "category": "logs_and_checkpoints",
        "priority": "medium",
        "title_tr": "Runtime logu",
        "description_tr": "Notebook runtime boyunca yazilan metin logu.",
        "reader_goal": "Calisma sirasindaki log ciktilarini okumak",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 81,
    },
    {
        "path": REPO_TELEMETRY_DIR / "latest_status.json",
        "category": "logs_and_checkpoints",
        "priority": "low",
        "title_tr": "Son durum ozeti",
        "description_tr": "Notebook'un son durum snapshot'i.",
        "reader_goal": "Kosunun son durumunu hizli kontrol etmek",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 82,
    },
    {
        "path": REPO_TELEMETRY_DIR / "summary.json",
        "category": "logs_and_checkpoints",
        "priority": "high",
        "title_tr": "Telemetry ozeti",
        "description_tr": "Notebook final ozet dosyasi.",
        "reader_goal": "Notebook final ozetini okumak",
        "generated_by": "notebook_2",
        "decision_importance": "run_overview",
        "read_order": 83,
    },
    {
        "path": REPO_CHECKPOINT_STATE_DIR / "best_checkpoint.json",
        "category": "logs_and_checkpoints",
        "priority": "medium",
        "title_tr": "Best checkpoint manifesti",
        "description_tr": "En iyi checkpoint'in repo mirror manifesti.",
        "reader_goal": "Hangi checkpoint secildigini gormek",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 84,
    },
    {
        "path": REPO_CHECKPOINT_STATE_DIR / "latest_checkpoint.json",
        "category": "logs_and_checkpoints",
        "priority": "medium",
        "title_tr": "Latest checkpoint manifesti",
        "description_tr": "Son checkpoint manifesti.",
        "reader_goal": "Checkpoint akisini gormek",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 85,
    },
    {
        "path": REPO_CHECKPOINT_STATE_DIR / "checkpoint_index.json",
        "category": "logs_and_checkpoints",
        "priority": "low",
        "title_tr": "Checkpoint indexi",
        "description_tr": "Mirror edilen checkpoint manifest listesi.",
        "reader_goal": "Checkpoint kayitlarini toplu gormek",
        "generated_by": "notebook_2",
        "decision_importance": "runtime_diagnostic",
        "read_order": 86,
    },
]
summary_payload = merge_training_summary_fields(
    root=ROOT,
    telemetry=TELEMETRY,
    payload={
        "recommendation": {
            "decision": str(STATE.get("recommendation_decision") or "pending"),
            "change_count": int((STATE.get("recommendation_report") or {}).get("change_count", 0)),
            "manual_overrides": dict(MANUAL_PARAM_OVERRIDES or {}),
        },
        "run_id": RUN_ID,
        "run_label": RUN_ID,
        "crop_name": CROP_NAME,
        "part_name": PART_NAME,
        "notebook_surface": "2_train_continual_sd_lora_adapter.ipynb",
        "dataset_roots": {
            "runtime_dataset_root": RUNTIME_DATASET_ROOT,
            "runtime_dataset_key": str(STATE.get("runtime_dataset_key") or ""),
            "selected_dataset_name": str(STATE.get("selected_dataset_name") or ""),
            "selected_runtime_dataset_root": str(STATE.get("selected_dataset_root") or ""),
            "resolved_ood_root": str(STATE.get("resolved_ood_root") or ""),
            "resolved_runtime_dataset_root": str(STATE.get("runtime_dataset_root") or ""),
            "dataset_lineage": dataset_lineage,
        },
        "notebook_parameters": {
            "epochs": (STATE.get("effective_params") or {}).get("EPOCHS"),
            "batch_size": (STATE.get("effective_params") or {}).get("BATCH_SIZE"),
            "learning_rate": (STATE.get("effective_params") or {}).get("LEARNING_RATE"),
            "lora_r": (STATE.get("effective_params") or {}).get("LORA_R"),
            "ood_factor": (STATE.get("effective_params") or {}).get("OOD_FACTOR"),
            "mixed_precision": (STATE.get("effective_params") or {}).get("MIXED_PRECISION"),
            "num_workers": (STATE.get("effective_params") or {}).get("NUM_WORKERS"),
            "checkpoint_every_n_steps": (STATE.get("effective_params") or {}).get("CHECKPOINT_EVERY_N_STEPS"),
            "enable_bayesian_optimization": bool(
                (STATE.get("effective_params") or {}).get("ENABLE_BAYESIAN_OPTIMIZATION", True)
            ),
        },
        "export_paths": {
            "repo_run_dir": str(REPO_RUN_DIR),
            "repo_output_dir": str(REPO_OUTPUT_DIR),
            "repo_telemetry_dir": str(REPO_TELEMETRY_DIR),
            "repo_checkpoint_state_dir": str(REPO_CHECKPOINT_STATE_DIR),
            "executed_notebook_path": str(notebook_export_result or REPO_NOTEBOOK_OUTPUT_PATH),
            "adapter_export_dir": str(STATE.get("adapter_export_dir") or ""),
        },
        "access_check": STATE.get("access_report", {}),
        "behavioral_acceptance_summary": {
            "status": (STATE.get("behavioral_acceptance") or {}).get("status"),
            "pass": (STATE.get("behavioral_acceptance") or {}).get("pass"),
            "decision_authority": "adapter_behavioral_acceptance",
        },
        "production_readiness_compatibility": STATE.get("production_readiness") or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
    },
    extra_entries=extra_entries,
)
TELEMETRY.merge_summary_metadata(
    {
        "access_check": STATE.get("access_report", {}),
        "repo_paths": {
            "repo_run_dir": str(REPO_RUN_DIR),
            "repo_output_dir": str(REPO_OUTPUT_DIR),
            "repo_telemetry_dir": str(REPO_TELEMETRY_DIR),
            "repo_checkpoint_state_dir": str(REPO_CHECKPOINT_STATE_DIR),
        },
        "training_summary": summary_payload,
    }
)
TELEMETRY.close(
    {
        "status": "ok",
        "evaluation_splits": sorted((STATE.get("evaluation_artifacts") or {}).keys()),
        "cell_outputs_dir": str(TELEMETRY.artifacts_dir / "cell_outputs"),
        "repo_run_dir": str(REPO_RUN_DIR),
        "run_label": RUN_ID,
    }
)
REPO_RUN_EXPORTS = save_run_outputs_to_repo()
for key in sorted(REPO_RUN_EXPORTS):
    print(f"[RUNS] {key} -> {REPO_RUN_EXPORTS[key]}")
print(f"[RUNS] notebook -> {REPO_NOTEBOOK_OUTPUT_PATH}")
effective_params = dict(STATE.get("effective_params") or {})
if bool(effective_params.get("AUTO_PUSH_TO_GITHUB", AUTO_PUSH_TO_GITHUB)):
    try:
        git_push_report = push_repo_run_to_github(
            ROOT,
            RUN_ID,
            run_relative_dir=REPO_RUN_DIR.relative_to(ROOT),
            remote_name=effective_params.get("AUTO_PUSH_REMOTE_NAME", AUTO_PUSH_REMOTE_NAME),
            branch=effective_params.get("AUTO_PUSH_BRANCH", AUTO_PUSH_BRANCH),
            print_fn=print,
        )
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[GIT] Auto-push skipped: {exc}")
        git_push_report = {"enabled": True, "pushed": False, "run_dir": str(REPO_RUN_DIR), "error": str(exc)}
else:
    git_push_report = {"enabled": False, "pushed": False, "run_dir": str(REPO_RUN_DIR)}
STATE["git_push_report"] = git_push_report
disconnect_report = build_notebook_completion_report(
    state=STATE,
    telemetry=TELEMETRY,
    repo_run_exports=REPO_RUN_EXPORTS,
    notebook_export_path=notebook_export_result or REPO_NOTEBOOK_OUTPUT_PATH,
)
STATE["auto_disconnect_report"] = disconnect_report
print(f"[COLAB] completion checks -> {disconnect_report['checks']}")
maybe_auto_disconnect_colab_runtime(
    enabled=bool(effective_params.get("AUTO_DISCONNECT_RUNTIME", AUTO_DISCONNECT_RUNTIME)),
    grace_period_sec=int(effective_params.get("AUTO_DISCONNECT_GRACE_SECONDS", AUTO_DISCONNECT_GRACE_SECONDS)),
    telemetry=TELEMETRY,
    completion_report=disconnect_report,
)
