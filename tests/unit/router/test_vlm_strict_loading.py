from unittest.mock import patch

import pytest
import torch

from src.router.router_pipeline import RouterPipeline
from src.router.vlm_stages import check_vlm_dependencies


def _fake_import(transformers_version, *, open_clip_available=True):
    def fake_import(name):
        if name == 'transformers':
            class _T:
                __version__ = transformers_version
            return _T
        if name == 'open_clip' and open_clip_available:
            return object()
        raise ImportError(name)

    return fake_import


def test_check_vlm_dependencies_reports_old_transformers_and_missing_optional():
    diagnostics = check_vlm_dependencies(import_module=_fake_import('4.40.0', open_clip_available=False))

    assert "outside the repo's validated range" in diagnostics['transformers_warning']
    assert '5.1.0' in diagnostics['transformers_warning']
    assert diagnostics['missing_deps'] == ['open-clip-torch']
    assert diagnostics['install_command'] == '!pip install open-clip-torch'


def test_check_vlm_dependencies_ok_path():
    diagnostics = check_vlm_dependencies(import_module=_fake_import('5.1.0'))

    assert diagnostics['transformers_warning'] is None
    assert diagnostics['missing_deps'] == []
    assert diagnostics['install_command'] is None


@pytest.mark.parametrize('version', ['5.2.0', '6.0.0'])
def test_check_vlm_dependencies_warns_outside_validated_range(version):
    diagnostics = check_vlm_dependencies(import_module=_fake_import(version))

    assert "outside the repo's validated range" in diagnostics['transformers_warning']
    assert '5.1.0' in diagnostics['transformers_warning']


class _FakeProcessor:
    def __call__(self, images, return_tensors='pt'):
        return {'pixel_values': torch.randn(1, 3, 224, 224)}


class _FakeModelOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _FakeModel:
    def __init__(self, logits: torch.Tensor, id2label=None):
        self._logits = logits
        self.config = type('Config', (), {'id2label': id2label or {}})()

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        return _FakeModelOutput(self._logits)


def test_strict_loading_requires_model_ids():
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': True,
        'router': {
            'vlm': {
                'enabled': True,
                'strict_model_loading': True,
                'model_source': 'huggingface',
                'model_ids': {
                    'sam': 'invalid/sam',
                    'bioclip': 'invalid/bioclip'
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = RouterPipeline(config=config, device='cpu')

    pipeline._load_sam = lambda model_id: (_ for _ in ()).throw(ValueError('bad sam id'))

    with pytest.raises(RuntimeError, match='Strict VLM model loading failed'):
        pipeline.load_models()


def test_non_strict_loading_falls_back_to_placeholder():
    """Test that non-strict mode leaves models as None instead of using placeholder strings."""
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': False,
        'router': {
            'vlm': {
                'enabled': True,
                'strict_model_loading': False,
                'model_source': 'huggingface',
                'model_ids': {
                    'sam': 'invalid/sam',
                    'bioclip': 'invalid/bioclip'
                }
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = RouterPipeline(config=config, device='cpu')

    pipeline._load_sam = lambda model_id: (_ for _ in ()).throw(ValueError('bad sam id'))

    pipeline.load_models()

    # Models should remain None (not placeholder strings) - inspired by reference implementation
    assert pipeline.models_loaded is False
    assert pipeline.sam_model is None
    assert pipeline.bioclip is None


def test_analyzer_resolution_is_sam3_only():
    config = {
        'vlm_enabled': True,
        'router': {
            'vlm': {
                'enabled': True,
            }
        }
    }

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = RouterPipeline(config=config, device='cpu')

    pipeline.actual_pipeline = 'dino'
    assert pipeline._resolve_analyzer_for_active_pipeline() is None

    pipeline.actual_pipeline = 'sam3'
    assert pipeline._resolve_analyzer_for_active_pipeline() == pipeline._analyze_image_sam3


def test_strict_loading_with_models_runs_inference():
    config = {
        'vlm_enabled': True,
        'vlm_strict_model_loading': True,
        'router': {
            'crop_mapping': {
                'tomato': {'parts': ['leaf', 'whole']},
                'potato': {'parts': ['leaf', 'tuber']}
            },
            'vlm': {
                'enabled': True,
                'strict_model_loading': True,
                'model_source': 'huggingface',
                'model_ids': {
                    'sam': 'fake-sam',
                    'bioclip': 'fake-bioclip'
                },
                'crop_labels': ['tomato', 'potato'],
                'part_labels': ['leaf', 'whole']
            }
        }
    }

    fake_model = _FakeModel(torch.tensor([[5.0, 1.0]]), id2label={0: 'tomato', 1: 'potato'})
    fake_processor = _FakeProcessor()

    with patch('torch.cuda.is_available', return_value=False):
        pipeline = RouterPipeline(config=config, device='cpu')

    pipeline._load_sam = lambda model_id: (fake_processor, fake_model)
    pipeline._load_clip_like_model = lambda model_id: (fake_processor, fake_model)
    pipeline.load_models()

    pipeline._analyze_image_sam3 = lambda pil_image, image_size, confidence_threshold=0.8, max_detections=None: {
        'detections': [
            {
                'crop': 'tomato',
                'part': 'leaf',
                'crop_confidence': 0.95,
                'part_confidence': 0.90,
                'bbox': [0.0, 0.0, 100.0, 100.0],
            }
        ],
        'image_size': image_size,
        'processing_time_ms': 1.0,
    }

    result = pipeline.analyze_image(torch.rand(3, 224, 224))
    detection = result['detections'][0]

    assert pipeline.is_ready() is True
    assert detection['crop'] == 'tomato'
    assert detection['part'] == 'leaf'
    assert detection['crop_confidence'] > 0.9
