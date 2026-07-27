# Auto-extracted from colab_notebooks/3_validate_exported_adapter_directly.ipynb cell 5.
# Keep notebook execute-only cells thin; edit behavior here.

if ADAPTER_DIR is None:
    selected_index = adapter_selector.value if adapter_selector is not None else SELECTED_ADAPTER_INDEX
    selected_candidate = adapter_candidates[int(selected_index)]
    ADAPTER_DIR = selected_candidate['adapter_dir']
    if CROP_NAME is None:
        CROP_NAME = selected_candidate.get('crop_name')
    print(json.dumps(selected_candidate, indent=2))
    FORCE_ADAPTER_RESCAN = False

summary = load_adapter_summary(
    CROP_NAME,
    adapter_dir=ADAPTER_DIR,
    adapter_root=ADAPTER_ROOT,
    config_env=CONFIG_ENV,
    device=DEVICE,
)

CROP_NAME = summary['crop_name']
ADAPTER_DIR = summary['resolved_adapter_dir']
print(f"Cozulen crop_name={CROP_NAME}")
print(f"Cozulen adapter_dir={ADAPTER_DIR}")

print(json.dumps(summary, indent=2))
