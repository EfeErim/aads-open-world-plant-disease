# Auto-extracted from colab_notebooks/2_train_continual_sd_lora_adapter.ipynb cell 7.
# Keep notebook execute-only cells thin; edit behavior here.

with TELEMETRY.capture_cell_output("Cell 5: Engine Init"):
    from scripts.colab_dataset_layout import resolve_notebook_training_classes
    from src.adapter.independent_crop_adapter import IndependentCropAdapter
    from src.data.loaders import create_training_loaders

    def _normalize(name: str) -> str:
        normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized.strip("_")

    if not STATE.get("validated"):
        raise RuntimeError("Once dataset validation hucresini calistirin.")

    effective_params = dict(STATE.get("effective_params") or {})
    if not effective_params:
        raise RuntimeError("Effective notebook parameters cozulmedi. Once dataset validation hucresini yeniden calistirin.")

    crop_name = CROP_NAME.strip().lower()
    dataset_key = str(STATE.get("runtime_dataset_key") or DATASET_NAME).strip()
    if not dataset_key:
        raise RuntimeError("Runtime dataset key cozulmedi. Once dataset validation hucresini calistirin.")
    runtime_data_root = Path(STATE.get("runtime_dataset_root") or RUNTIME_DATASET_ROOT).expanduser()
    if not runtime_data_root.is_absolute():
        runtime_data_root = (ROOT / runtime_data_root).resolve()
    class_root = runtime_data_root / dataset_key / "continual"
    if not class_root.is_dir():
        raise RuntimeError(f"Prepared runtime continual split not found: {class_root}")

    from scripts.colab_notebook_helpers import (
        apply_notebook_optimization_proposal,
        print_notebook_optimization_campaign_status,
        resolve_notebook_optimization_campaign,
        summarize_notebook_optimization_campaign,
    )

    optimization_campaign_mode = "continue" if bool(effective_params.get("ENABLE_BAYESIAN_OPTIMIZATION", True)) else "disabled"
    backbone_model_name = str(dict(BASE_CONFIG.get("training", {}).get("continual", {})).get("backbone", {}).get("model_name", ""))
    optimization_campaign = resolve_notebook_optimization_campaign(
        root=ROOT,
        runtime_dataset_root=runtime_data_root / dataset_key,
        dataset_key=dataset_key,
        crop_name=CROP_NAME,
        part_name=PART_NAME,
        backbone_model_name=backbone_model_name,
        notebook_parameters=effective_params,
        mode=optimization_campaign_mode,
        telemetry=TELEMETRY,
    )
    print_notebook_optimization_campaign_status(optimization_campaign, print_fn=print)
    proposal_application = apply_notebook_optimization_proposal(
        notebook_parameters=effective_params,
        campaign=optimization_campaign,
        telemetry=TELEMETRY,
        print_fn=print,
    )
    optimization_campaign = proposal_application.get("campaign", optimization_campaign)
    effective_params = dict(proposal_application.get("notebook_parameters", effective_params))
    STATE["effective_params"] = dict(effective_params)
    STATE["notebook_parameters"] = dict(effective_params)
    STATE["optimization_campaign"] = optimization_campaign
    STATE["optimization_applied_params"] = proposal_application
    STATE["recommendation_report"] = summarize_notebook_optimization_campaign(optimization_campaign)
    STATE["recommendation_decision"] = str(optimization_campaign.get("status", "disabled"))
    if not proposal_application.get("applied"):
        if optimization_campaign.get("status") == "bootstrap_pending":
            print("[OPT] No prior comparable run yet. Bootstrap run will use the visible notebook parameters.")
        elif optimization_campaign.get("status") == "stopped":
            print("[OPT] Campaign is marked stopped. Notebook will keep the visible parameters as-is.")
        elif optimization_campaign.get("status") == "active" and not optimization_campaign.get("next_proposal"):
            print("[OPT] No unseen proposal is available. Notebook will keep the visible parameters as-is.")

    raw_class_names = sorted(path.name for path in class_root.iterdir() if path.is_dir())
    normalized_to_raw = {}
    for raw_name in raw_class_names:
        normalized_name = _normalize(raw_name)
        if not normalized_name:
            continue
        existing_raw = normalized_to_raw.get(normalized_name)
        if existing_raw is not None and existing_raw != raw_name:
            raise RuntimeError(
                f"Class names collapse to the same normalized key: {existing_raw!r} and {raw_name!r}"
            )
        normalized_to_raw[normalized_name] = raw_name
    available = sorted(normalized_to_raw)
    resolved_ood_root = str(STATE.get("resolved_ood_root") or "").strip()
    if resolved_ood_root:
        print(f"[OOD] selected ood root={resolved_ood_root}")
    resolved_oe_root = str(STATE.get("resolved_oe_root") or "").strip()
    if resolved_oe_root:
        print(f"[OE] selected oe root={resolved_oe_root}")

    class_resolution = resolve_notebook_training_classes(
        available_classes=available,
        crop_name=crop_name,
        taxonomy_path=ROOT / "config" / "plant_taxonomy.json",
    )
    aligned = list(class_resolution.get("selected_classes", available))

    if not aligned:
        raise RuntimeError(f"No usable classes for crop '{crop_name}'. Available: {available}")

    final_class_names = [normalized_to_raw[name] for name in aligned if name in normalized_to_raw]
    if len(final_class_names) != len(aligned):
        missing = sorted(set(aligned) - set(normalized_to_raw))
        raise RuntimeError(f"Resolved class names are not present on disk after normalization: {missing}")
    STATE["runtime_dataset_root"] = runtime_data_root
    STATE["runtime_dataset_key"] = dataset_key

    STATE["class_names"] = final_class_names
    STATE["class_resolution"] = class_resolution
    print(f"[CLASSES] {final_class_names}")
    print(
        f"[CLASSES] mode={'taxonomy_filter' if class_resolution.get('used_taxonomy_filter') else 'dataset_fallback'} "
        f"reason={class_resolution.get('reason', 'unknown')} "
        f"matched={len(class_resolution.get('matched_classes', []))} "
        f"unmatched={len(class_resolution.get('unmatched_classes', []))}"
    )
    if class_resolution.get("unmatched_classes"):
        print(f"[CLASSES] taxonomy-unmatched classes kept: {class_resolution['unmatched_classes']}")

    continual_cfg = json.loads(json.dumps(BASE_CONFIG.get("training", {}).get("continual", {})))
    continual_cfg["device"] = DEVICE
    continual_cfg["num_epochs"] = int(effective_params["EPOCHS"])
    continual_cfg["batch_size"] = int(effective_params["BATCH_SIZE"])
    continual_cfg["learning_rate"] = float(effective_params["LEARNING_RATE"])
    continual_cfg["adapter"]["lora_r"] = int(effective_params["LORA_R"])
    continual_cfg["adapter"]["lora_alpha"] = int(effective_params["LORA_ALPHA"])
    continual_cfg["adapter"]["lora_dropout"] = float(effective_params["LORA_DROPOUT"])
    continual_cfg.setdefault("fusion", {})["dropout"] = float(effective_params.get("FUSION_DROPOUT", 0.1))
    continual_cfg["weight_decay"] = float(effective_params["WEIGHT_DECAY"])
    continual_cfg["deterministic"] = bool(effective_params["DETERMINISTIC"])
    data_cfg = continual_cfg.setdefault("data", {})
    data_cfg["augmentation_policy"] = str(effective_params.get("AUGMENTATION_POLICY", AUGMENTATION_POLICY))
    data_cfg["randaugment_num_ops"] = int(effective_params["RANDAUGMENT_NUM_OPS"])
    data_cfg["randaugment_magnitude"] = int(effective_params["RANDAUGMENT_MAGNITUDE"])
    data_cfg["allow_under_min_training"] = bool(effective_params["ALLOW_UNDER_MIN_TRAINING"])
    continual_cfg["ood"]["threshold_factor"] = float(effective_params["OOD_FACTOR"])
    continual_cfg["ood"]["react_enabled"] = bool(effective_params.get("REACT_ENABLED", continual_cfg["ood"].get("react_enabled", False)))
    continual_cfg["ood"]["react_percentile"] = float(
        effective_params.get("REACT_PERCENTILE", continual_cfg["ood"].get("react_percentile", 0.99))
    )
    continual_cfg["ood"]["sure_semantic_percentile"] = float(effective_params["SURE_SEMANTIC_PERCENTILE"])
    continual_cfg["ood"]["sure_confidence_percentile"] = float(effective_params["SURE_CONFIDENCE_PERCENTILE"])
    continual_cfg["ood"]["conformal_alpha"] = float(effective_params["CONFORMAL_ALPHA"])
    continual_cfg["ood"]["conformal_method"] = str(effective_params["CONFORMAL_METHOD"])
    continual_cfg["ood"]["conformal_raps_lambda"] = float(effective_params["CONFORMAL_RAPS_LAMBDA"])
    continual_cfg["ood"]["conformal_raps_k_reg"] = int(effective_params["CONFORMAL_RAPS_K_REG"])
    continual_cfg["ood"]["ber_enabled"] = bool(effective_params["BER_ENABLED"])
    continual_cfg["ood"]["ber_lambda_old"] = float(effective_params["BER_LAMBDA_OLD"])
    continual_cfg["ood"]["ber_lambda_new"] = float(effective_params["BER_LAMBDA_NEW"])
    continual_cfg["ood"]["ber_warmup_steps"] = int(effective_params["BER_WARMUP_STEPS"])
    continual_cfg["ood"]["oe_enabled"] = bool(effective_params["OE_ENABLED"])
    continual_cfg["ood"]["oe_loss_weight"] = float(effective_params.get("OE_LOSS_WEIGHT", OE_LOSS_WEIGHT))
    continual_cfg["ood"]["oe_root"] = resolved_oe_root
    print(f"[ENGINE][OOD_CFG] {json.dumps(continual_cfg['ood'], sort_keys=True)}")

    classifier_rebalance_cfg = continual_cfg.setdefault("classifier_rebalance", {})
    classifier_rebalance_cfg["enabled"] = bool(
        effective_params.get("CLASSIFIER_REBALANCE_ENABLED", classifier_rebalance_cfg.get("enabled", False))
    )
    classifier_rebalance_cfg["logit_adjustment_tau"] = float(
        effective_params.get(
            "CLASSIFIER_REBALANCE_LOGIT_ADJUSTMENT_TAU",
            classifier_rebalance_cfg.get("logit_adjustment_tau", 1.0),
        )
    )

    optimization_cfg = continual_cfg.setdefault("optimization", {})
    optimization_cfg["grad_accumulation_steps"] = int(effective_params["GRAD_ACCUM_STEPS"])
    optimization_cfg["max_grad_norm"] = float(effective_params["MAX_GRAD_NORM"])
    optimization_cfg["mixed_precision"] = str(effective_params["MIXED_PRECISION"])
    optimization_cfg["label_smoothing"] = float(effective_params["LABEL_SMOOTHING"])
    optimization_cfg["loss_name"] = str(effective_params["LOSS_NAME"]).strip().lower()
    optimization_cfg["logitnorm_tau"] = float(effective_params["LOGITNORM_TAU"])
    scheduler_cfg = optimization_cfg.setdefault("scheduler", {})
    scheduler_cfg["name"] = str(effective_params["SCHEDULER_NAME"])
    scheduler_cfg["warmup_ratio"] = float(effective_params["SCHEDULER_WARMUP_RATIO"])
    scheduler_cfg["min_lr"] = float(effective_params["SCHEDULER_MIN_LR"])
    scheduler_cfg["step_on"] = str(scheduler_cfg.get("step_on", "batch"))
    early_stopping_cfg = continual_cfg.setdefault("early_stopping", {})
    early_stopping_cfg["enabled"] = True
    early_stopping_cfg["patience"] = int(effective_params["EARLY_STOPPING_PATIENCE"])
    early_stopping_cfg["min_delta"] = float(effective_params["EARLY_STOPPING_MIN_DELTA"])
    print(
        f"[ENGINE][OPT_CFG] loss={optimization_cfg['loss_name']} tau={optimization_cfg['logitnorm_tau']} "
        f"label_smoothing={optimization_cfg['label_smoothing']} mixed_precision={optimization_cfg['mixed_precision']}"
    )

    adapter = IndependentCropAdapter(crop_name=crop_name, device=DEVICE)
    if hasattr(adapter, "part_name"):
        adapter.part_name = str(PART_NAME or "unspecified").strip().lower() or "unspecified"
    print("[ENGINE] Initializing adapter (may download backbone)...")
    adapter.initialize_engine(class_names=STATE["class_names"], config={"training": {"continual": continual_cfg}})

    loader_kwargs = {}
    if int(effective_params["NUM_WORKERS"]) > 0:
        loader_kwargs["prefetch_factor"] = int(effective_params["PREFETCH"])

    loaders = create_training_loaders(
        data_dir=str(runtime_data_root),
        crop=dataset_key,
        class_names=final_class_names,
        batch_size=int(effective_params["BATCH_SIZE"]),
        num_workers=int(effective_params["NUM_WORKERS"]),
        use_cache=bool(effective_params["USE_CACHE"]),
        cache_size=CACHE_SIZE,
        cache_train_split=bool(effective_params["CACHE_TRAIN_SPLIT"]),
        target_size=TARGET_SIZE,
        error_policy=LOADER_ERROR_POLICY,
        sampler=DATA_SAMPLER,
        seed=int(effective_params["SEED"]),
        validate_images_on_init=VALIDATE_IMAGES_ON_INIT,
        augmentation_policy=str(effective_params.get("AUGMENTATION_POLICY", AUGMENTATION_POLICY)),
        randaugment_num_ops=int(effective_params["RANDAUGMENT_NUM_OPS"]),
        randaugment_magnitude=int(effective_params["RANDAUGMENT_MAGNITUDE"]),
        ood_root=resolved_ood_root or None,
        oe_root=resolved_oe_root or None,
        pin_memory=bool(effective_params["PIN_MEMORY"]),
        **loader_kwargs,
    )
    train_dataset_class_to_idx = dict(getattr(getattr(loaders.get("train"), "dataset", None), "class_to_idx", {}) or {})
    adapter_class_to_idx = dict(getattr(adapter, "class_to_idx", {}) or {})
    if train_dataset_class_to_idx and adapter_class_to_idx != train_dataset_class_to_idx:
        raise RuntimeError(
            "Adapter class_to_idx does not match the runtime dataset loader class_to_idx. "
            f"adapter={adapter_class_to_idx} loader={train_dataset_class_to_idx}"
        )

    STATE["crop_name"] = crop_name
    STATE["part_name"] = str(PART_NAME or "unspecified").strip().lower() or "unspecified"
    STATE["adapter"] = adapter
    STATE["loaders"] = loaders
    STATE["continual_config"] = continual_cfg

    trainable = sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
    print(f"[ENGINE] Hazir. trainable_params={trainable:,}  classes={len(final_class_names)}")
    TELEMETRY.update_latest(
        {
            "phase": "engine_ready",
            "class_count": len(final_class_names),
            "runtime_dataset_root": str(runtime_data_root),
            "runtime_dataset_key": dataset_key,
            "selected_dataset_name": str(STATE.get("selected_dataset_name") or ""),
            "resolved_ood_root": str(STATE.get("resolved_ood_root") or ""),
            "resolved_oe_root": str(STATE.get("resolved_oe_root") or ""),
            "effective_batch_size": int(effective_params["BATCH_SIZE"]) * int(effective_params["GRAD_ACCUM_STEPS"]),
        }
    )
