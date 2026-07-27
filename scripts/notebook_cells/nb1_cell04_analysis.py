# Auto-extracted from colab_notebooks/1_identify_crop_part_with_router.ipynb cell 6.
# Keep notebook execute-only cells thin; edit behavior here.

import json

from scripts.colab_repo_bootstrap import running_in_colab
from scripts.colab_router_adapter_inference import run_inference

UPLOAD_NEW_IMAGE = ANALYSIS_IMAGE_PATH is None and FORCE_UPLOAD_IF_NO_IMAGE

if ANALYSIS_IMAGE_PATH is None and UPLOAD_NEW_IMAGE:
    if not running_in_colab():
        raise ValueError('Set ANALYSIS_IMAGE_PATH manually when running outside Colab.')
    from google.colab import files
    uploaded = files.upload()
    ANALYSIS_IMAGE_PATH = next(iter(uploaded.keys()))

if ANALYSIS_IMAGE_PATH is None:
    raise ValueError('ANALYSIS_IMAGE_PATH is required when upload is disabled.')

# Router models are cached, so rerun this cell to try another image.
result = run_inference(
    ANALYSIS_IMAGE_PATH,
    config_env=CONFIG_ENV,
    crop_hint=CROP_HINT,
    part_hint=PART_HINT,
    device=DEVICE,
    status_printer=print,
    include_diagnostics=SHOW_ROUTER_DIAGNOSTICS,
    top_candidates=TOP_CROP_CANDIDATES,
    runtime_profile=ROUTER_RUNTIME_PROFILE,
    max_image_side=MAX_IMAGE_SIDE_FOR_ROUTER,
    include_adapter_target=INCLUDE_ADAPTER_TARGET,
)

runtime_profile = str(result.get('runtime_profile', '') or '')
if runtime_profile:
    print(f'[ROUTER] active_runtime_profile={runtime_profile}')

adapter_target = dict(result.get('adapter_target') or {})
if INCLUDE_ADAPTER_TARGET and adapter_target.get('crop'):
    print(
        f"[ADAPTER] target_crop={adapter_target.get('crop')} "
        f"adapter_dir={adapter_target.get('adapter_dir')} "
        f"exists={bool(adapter_target.get('exists', False))}"
    )

diagnostics = dict(result.get('diagnostics') or {})
top_candidates = list(diagnostics.get('top_crop_candidates') or [])
if top_candidates:
    print('[DIAGNOSTICS] Top crop candidates:')
    for index, candidate in enumerate(top_candidates, start=1):
        crop_name = str(candidate.get('crop', 'unknown') or 'unknown')
        part_name = str(candidate.get('part', 'unknown') or 'unknown')
        crop_conf = float(candidate.get('crop_confidence', 0.0) or 0.0)
        part_conf = float(candidate.get('part_confidence', 0.0) or 0.0)
        print(
            f"  {index}. crop={crop_name} part={part_name} "
            f"crop_conf={crop_conf:.3f} part_conf={part_conf:.3f}"
        )

if diagnostics:
    print(
        f"[DIAGNOSTICS] crop_margin={float(diagnostics.get('crop_confidence_margin', 0.0) or 0.0):.3f} "
        f"raw_part_label={str(diagnostics.get('raw_part_label', '') or '')} "
        f"raw_part_conf={float(diagnostics.get('raw_part_confidence', 0.0) or 0.0):.3f} "
        f"unknown_conf={float(diagnostics.get('part_unknown_confidence', 0.0) or 0.0):.3f}"
    )
    rejection_reason = str(diagnostics.get('part_rejection_reason', '') or '').strip()
    if rejection_reason:
        print(f'[DIAGNOSTICS] part_rejection_reason={rejection_reason}')

primary_crop_conf = float(result.get('router_confidence', 0.0) or 0.0)
runner_up_crop_conf = None
if len(top_candidates) > 1:
    runner_up_crop_conf = float(top_candidates[1].get('crop_confidence', 0.0) or 0.0)
effective_margin = None if runner_up_crop_conf is None else (primary_crop_conf - runner_up_crop_conf)

notebook_gate_reasons = []
if ENFORCE_NOTEBOOK_ROUTER_GATE and not CROP_HINT:
    if primary_crop_conf < float(ROUTER_RESULT_MIN_CONFIDENCE):
        notebook_gate_reasons.append(
            f"crop_confidence={primary_crop_conf:.3f} < min_confidence={float(ROUTER_RESULT_MIN_CONFIDENCE):.3f}"
        )
    if effective_margin is not None and effective_margin < float(ROUTER_RESULT_MIN_MARGIN):
        notebook_gate_reasons.append(
            f"crop_margin={effective_margin:.3f} < min_margin={float(ROUTER_RESULT_MIN_MARGIN):.3f}"
        )

if notebook_gate_reasons:
    notebook_gate_message = 'Notebook router gate rejected prediction: ' + '; '.join(notebook_gate_reasons)
    print(f'[GATE] {notebook_gate_message}')
    result['status'] = 'uncertain'
    result['message'] = notebook_gate_message
    result['crop'] = 'unknown'
    result['part'] = 'unknown'
    result['adapter_target'] = {'crop': None, 'adapter_dir': None, 'exists': False}
    result['notebook_gate'] = {
        'applied': True,
        'accepted': False,
        'reasons': notebook_gate_reasons,
        'min_confidence': float(ROUTER_RESULT_MIN_CONFIDENCE),
        'min_margin': float(ROUTER_RESULT_MIN_MARGIN),
    }
elif ENFORCE_NOTEBOOK_ROUTER_GATE:
    result['notebook_gate'] = {
        'applied': True,
        'accepted': True,
        'reasons': [],
        'min_confidence': float(ROUTER_RESULT_MIN_CONFIDENCE),
        'min_margin': float(ROUTER_RESULT_MIN_MARGIN),
    }

router_details = dict(result.get('router_details') or {})
detections = list(router_details.get('detections') or [])
box_count = 0

if RENDER_ROUTER_VISUALIZATION:
    # Import only when visualization is enabled to reduce startup overhead.
    from PIL import Image, ImageDraw

    analysis_image = Image.open(ANALYSIS_IMAGE_PATH).convert('RGB')
    annotated_image = analysis_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    width, height = annotated_image.size

    for index, detection in enumerate(detections, start=1):
        if MAX_RENDERED_BOXES > 0 and index > MAX_RENDERED_BOXES:
            break
        bbox = detection.get('bbox')
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            x1 *= width
            x2 *= width
            y1 *= height
            y2 *= height
        x1 = max(0.0, min(width - 1.0, x1))
        y1 = max(0.0, min(height - 1.0, y1))
        x2 = max(0.0, min(width - 1.0, x2))
        y2 = max(0.0, min(height - 1.0, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        box_count += 1
        color = (255, 99, 71) if index == 1 else (66, 133, 244)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = f"{index}: {str(detection.get('crop', 'unknown') or 'unknown')} / {str(detection.get('part', 'unknown') or 'unknown')}"
        text_x = x1 + 4
        text_y = max(0.0, y1 - 18)
        draw.rectangle([text_x - 2, text_y - 2, text_x + 230, text_y + 16], fill=(0, 0, 0))
        draw.text((text_x, text_y), label, fill=(255, 255, 255))

    if box_count <= 0:
        print('[DIAGNOSTICS] No router detection boxes were found; showing only the raw image.')

    # Defer matplotlib import until visualization is needed.
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2 if box_count > 0 else 1, figsize=(16, 8))
    if box_count > 0:
        axes[0].imshow(analysis_image)
        axes[0].set_title('Orijinal')
        axes[0].axis('off')
        axes[1].imshow(annotated_image)
        axes[1].set_title('Router kutulari')
        axes[1].axis('off')
    else:
        axes.imshow(analysis_image)
        axes.set_title('Orijinal')
        axes.axis('off')
    plt.tight_layout()
else:
    print('[SPEED] Skipped box drawing and matplotlib rendering because RENDER_ROUTER_VISUALIZATION=False.')

if PRINT_JSON_RESULT:
    print(json.dumps(result, indent=2))
