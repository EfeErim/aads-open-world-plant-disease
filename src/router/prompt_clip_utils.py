from typing import Any, Dict, List, Optional, Tuple

import torch

TURKEY_PRIORITY_CROP_ALIASES: Dict[str, List[str]] = {
    'wheat': ['wheat', 'bread wheat', 'durum wheat', 'Triticum aestivum', 'Triticum durum'],
    'sugar beet': ['sugar beet', 'Beta vulgaris'],
    'tomato': ['tomato', 'Solanum lycopersicum'],
    'barley': ['barley', 'Hordeum vulgare'],
    'maize': ['maize', 'corn', 'Zea mays'],
    'potato': ['potato', 'Solanum tuberosum'],
    'grape': ['grape', 'grapevine', 'Vitis vinifera'],
    'strawberry': ['strawberry', 'Fragaria × ananassa'],
    'apricot': ['apricot', 'Prunus armeniaca'],
    'hazelnut': ['hazelnut', 'hazel', 'Corylus avellana'],
    'olive': ['olive', 'olive tree', 'Olea europaea'],
    'sunflower': ['sunflower', 'Helianthus annuus'],
    'cotton': ['cotton', 'Gossypium'],
    'apple': ['apple', 'apple tree', 'Malus domestica'],
}

TURKEY_PRIORITY_TOP_10_CROPS: Tuple[str, ...] = (
    'wheat',
    'maize',
    'barley',
    'sugar beet',
    'sunflower',
    'cotton',
    'tomato',
    'potato',
    'grape',
    'apple',
)

TURKEY_PRIORITY_CROP_PARTS: Dict[str, List[str]] = {
    'wheat': ['leaf', 'stem', 'root', 'grain', 'ear', 'spike', 'kernel', 'seedling', 'whole plant'],
    'sugar beet': ['leaf', 'root', 'crown', 'stem', 'seed', 'seedling', 'whole plant'],
    'tomato': ['leaf', 'fruit', 'flower', 'stem', 'seed', 'bud', 'shoot', 'seedling', 'whole plant'],
    'barley': ['leaf', 'stem', 'root', 'grain', 'ear', 'spike', 'kernel', 'seedling', 'whole plant'],
    'maize': ['leaf', 'stem', 'root', 'grain', 'ear', 'cob', 'kernel', 'tassel', 'husk', 'seedling', 'whole plant'],
    'potato': ['leaf', 'tuber', 'flower', 'stem', 'root', 'shoot', 'seedling', 'whole plant'],
    'grape': ['leaf', 'fruit', 'berry', 'vine', 'stem', 'tendril', 'shoot', 'whole plant'],
    'strawberry': ['leaf', 'fruit', 'berry', 'flower', 'crown', 'root', 'shoot', 'seedling', 'whole plant'],
    'apricot': ['leaf', 'fruit', 'flower', 'branch', 'stem', 'bark', 'trunk', 'seed', 'bud', 'shoot', 'whole plant'],
    'hazelnut': [
        'leaf', 'nut', 'kernel', 'shell', 'husk', 'involucre', 'cupule',
        'cluster', 'fruit cluster', 'nut cluster', 'hazelnut cluster',
        'flower', 'male catkin', 'female flower', 'catkin', 'branch', 'shoot',
        'stem', 'bark', 'trunk', 'root', 'bud', 'whole plant',
    ],
    'olive': ['leaf', 'fruit', 'flower', 'branch', 'stem', 'bark', 'trunk', 'seed', 'whole plant'],
    'sunflower': ['leaf', 'flower', 'head', 'stem', 'seed', 'root', 'bud', 'seedling', 'whole plant'],
    'cotton': ['leaf', 'flower', 'boll', 'stem', 'root', 'seed', 'bud', 'branch', 'whole plant'],
    'apple': ['leaf', 'fruit', 'flower', 'branch', 'bark', 'stem', 'whole plant', 'bud', 'seed'],
}

TURKEY_PRIORITY_PART_ALIASES: Dict[str, List[str]] = {
    'leaf': ['leaf', 'leaves', 'plant leaf', 'plant leaves', 'crop leaf', 'crop leaves'],
    'fruit': ['fruit', 'fruits', 'plant fruit', 'plant fruits', 'crop fruit', 'crop fruits'],
    'stem': ['stem', 'stems', 'plant stem', 'crop stem', 'shoot'],
    'flower': ['flower', 'flowers', 'plant flower', 'crop flower', 'bloom', 'blossom'],
    'root': ['root', 'roots', 'plant root', 'crop root'],
    'tuber': ['tuber', 'tubers', 'plant tuber', 'crop tuber'],
    'ear': ['ear', 'ears', 'grain ear', 'seed head'],
    'spike': ['spike', 'spikes', 'plant spike', 'grain spike'],
    'grain': ['grain', 'grains', 'seed', 'seeds', 'kernel', 'kernels'],
    'seed': ['seed', 'seeds', 'kernel', 'kernels', 'grain'],
    'berry': ['berry', 'berries', 'fruit berry'],
    'boll': ['boll', 'bolls', 'cotton boll'],
    'bud': ['bud', 'buds', 'flower bud', 'shoot bud'],
    'branch': ['branch', 'branches', 'twig'],
    'bark': ['bark', 'tree bark'],
    'whole plant': ['whole plant', 'whole', 'entire plant', 'plant'],
    'plant': ['plant', 'whole plant', 'entire plant'],
    'cob': ['cob', 'corn cob', 'maize cob'],
    'head': ['head', 'flower head', 'seed head'],
    'husk': ['husk', 'husks'],
    'shell': ['shell', 'shells'],
    'pod': ['pod', 'pods'],
    'crown': ['crown', 'plant crown'],
    'bulb': ['bulb', 'bulbs'],
    'catkin': ['catkin', 'catkins'],
    'kernel': ['kernel', 'kernels'],
    'tendril': ['tendril', 'tendrils'],
    'vine': ['vine', 'vines', 'grapevine'],
    'trunk': ['trunk', 'tree trunk'],
    'shoot': ['shoot', 'shoots', 'sprout'],
    'seedling': ['seedling', 'seedlings', 'young plant'],
}


def _crop_key(label: str) -> str:
    return str(label or '').strip().lower()


def _crop_alias_to_canonical() -> Dict[str, str]:
    canonical_by_alias: Dict[str, str] = {}
    for canonical_label, aliases in TURKEY_PRIORITY_CROP_ALIASES.items():
        canonical_key = _crop_key(canonical_label)
        if canonical_key:
            canonical_by_alias[canonical_key] = canonical_key
        for alias in aliases:
            alias_key = _crop_key(alias)
            if alias_key:
                canonical_by_alias[alias_key] = canonical_key
    return canonical_by_alias


def canonicalize_crop_prompt_label(label: str) -> str:
    """Resolve crop aliases to a maintained canonical label for prompt generation."""
    label_key = _crop_key(label)
    if not label_key:
        return ""
    return _crop_alias_to_canonical().get(label_key, label_key)


def default_prompt_templates_for_type(label_type: str) -> List[str]:
    """Return built-in prompt templates for a semantic label type."""
    if label_type == 'part':
        return [
            "a photo of a {term}",
            "a close-up photo of a {term}",
            "a macro photo of a {term}",
            "a detailed photo of a {term}",
            "a {term} on a plant",
            "a healthy {term}",
            "a diseased {term}",
            "a {term} with damage",
        ]
    if label_type == 'crop':
        return [
            "a photo of {term}",
            "a close-up photo of {term}",
            "a macro photo of {term}",
            "an image of {term}",
            "a {term} crop",
            "a {term} plant",
            "a {term} with disease",
            "a diseased {term}",
        ]
    return ["{term}"]


def _crop_part_prompt_variants(term: str, part: str) -> List[str]:
    return [
        f"a photo of a {term} {part}",
        f"a close-up photo of a {term} {part}",
        f"a diseased {term} {part}",
    ]


def _part_key(label: str) -> str:
    return str(label or '').strip().lower()


def part_prompt_aliases() -> Dict[str, List[str]]:
    """Return part prompt aliases used to broaden zero-shot organ coverage."""
    aliases = dict(TURKEY_PRIORITY_PART_ALIASES)
    return aliases


def get_prompt_templates_for_type(vlm_config: Dict[str, Any], label_type: str) -> List[str]:
    """Get dynamic prompt templates from config with sane default fallback."""
    templates_cfg = vlm_config.get('prompt_templates', {}) if isinstance(vlm_config, dict) else {}
    if not isinstance(templates_cfg, dict):
        templates_cfg = {}

    label_templates = templates_cfg.get(label_type, templates_cfg.get('default', []))
    if not isinstance(label_templates, list):
        label_templates = []

    cleaned_templates = [str(template).strip() for template in label_templates if str(template).strip()]
    return cleaned_templates if cleaned_templates else default_prompt_templates_for_type(label_type)


def crop_prompt_aliases() -> Dict[str, List[str]]:
    """Return crop prompt aliases including scientific names where available."""
    aliases = dict(TURKEY_PRIORITY_CROP_ALIASES)
    return aliases


def build_prompt_ensemble(label: str, label_type: str, vlm_config: Dict[str, Any]) -> List[str]:
    """Build multiple prompt variants for a single semantic class label."""
    label_text = str(label).strip()
    if not label_text:
        return []

    prompt_templates_cfg = vlm_config.get('prompt_templates', {}) if isinstance(vlm_config, dict) else {}
    custom_templates = prompt_templates_cfg.get(label_type, []) if isinstance(prompt_templates_cfg, dict) else []

    if label_type == 'part':
        aliases = part_prompt_aliases()
        alias_terms = aliases.get(_part_key(label_text), [label_text])
        base_terms = [term for term in [label_text, *alias_terms] if isinstance(term, str) and term.strip()]
        templates = custom_templates if custom_templates else default_prompt_templates_for_type(label_type)
    else:
        canonical_label = canonicalize_crop_prompt_label(label_text)
        aliases = crop_prompt_aliases()
        alias_terms = aliases.get(canonical_label, aliases.get(label_text.lower(), [label_text]))
        base_terms = [term for term in [canonical_label, label_text, *alias_terms] if isinstance(term, str) and term.strip()]

        templates = custom_templates if custom_templates else default_prompt_templates_for_type(label_type)

    prompts: List[str] = []
    seen = set()
    if label_type == 'crop' and not custom_templates:
        canonical_crop = canonicalize_crop_prompt_label(label_text)
        for part_name in TURKEY_PRIORITY_CROP_PARTS.get(canonical_crop or _crop_key(label_text), []):
            for prompt in _crop_part_prompt_variants(canonical_crop or label_text, part_name):
                key = prompt.lower()
                if key not in seen:
                    seen.add(key)
                    prompts.append(prompt)

    for term in base_terms:
        clean_term = term.strip()
        for template in templates:
            prompt = template.format(term=clean_term)
            key = prompt.lower()
            if key not in seen:
                seen.add(key)
                prompts.append(prompt)
    return prompts


def build_prompt_batch(labels: List[str], label_type: str, vlm_config: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    """Build prompt list and class index mapping for class-level aggregation."""
    prompt_texts: List[str] = []
    prompt_to_class: List[int] = []

    for class_index, label in enumerate(labels):
        class_prompts = build_prompt_ensemble(label, label_type=label_type, vlm_config=vlm_config)
        if not class_prompts:
            class_prompts = [str(label)]
        prompt_texts.extend(class_prompts)
        prompt_to_class.extend([class_index] * len(class_prompts))

    return prompt_texts, prompt_to_class


def open_set_unknown_prompts(label_type: str, known_labels: Optional[List[str]] = None) -> List[str]:
    """Prompt set for unknown/out-of-scope rejection."""
    _ = known_labels
    if label_type == 'part':
        return [
            "a photo of a rock or stone",
            "a photo of water or liquid",
            "a photo of a building or concrete",
            "a photo of an animal or insect",
        ]

    return [
        "a photo of a rock or stone",
        "a photo of water or liquid surface",
        "a photo of soil or dirt only",
        "a photo of a building or concrete",
        "a photo of an unrelated object",
    ]


def aggregate_prompt_logits(logits: torch.Tensor, prompt_to_class: List[int], num_classes: int) -> torch.Tensor:
    """Aggregate prompt-level logits into class-level logits using max pooling."""
    if logits.ndim == 2:
        logits = logits.squeeze(0)

    class_logits = torch.full(
        (num_classes,),
        float('-inf'),
        device=logits.device,
        dtype=logits.dtype,
    )

    for prompt_index, class_index in enumerate(prompt_to_class):
        class_logits[class_index] = torch.maximum(class_logits[class_index], logits[prompt_index])

    return class_logits.unsqueeze(0)


def get_clip_logit_scale(model: Any) -> float:
    """Get CLIP logit scale (temperature inverse) with safe fallback."""
    logit_scale_attr = getattr(model, 'logit_scale', None)
    if logit_scale_attr is None:
        return 1.0

    try:
        if torch.is_tensor(logit_scale_attr):
            scale_value = logit_scale_attr.detach().float().squeeze().exp().item()
        elif hasattr(logit_scale_attr, 'data') and torch.is_tensor(logit_scale_attr.data):
            scale_value = logit_scale_attr.data.detach().float().squeeze().exp().item()
        else:
            scale_value = float(logit_scale_attr)
        return max(1.0, min(scale_value, 100.0))
    except (TypeError, ValueError, OverflowError, RuntimeError, AttributeError):
        return 1.0
