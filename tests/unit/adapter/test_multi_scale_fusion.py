import pytest
import torch

from src.adapter.multi_scale_fusion import MultiScaleFeatureFusion, select_multiscale_features


def test_select_multiscale_features_clamps_indices():
    states = [torch.randn(2, 4, 8) for _ in range(3)]
    selected = select_multiscale_features(states, [0, 2, 10])
    assert len(selected) == 3
    assert selected[-1] is states[-1]


def test_select_multiscale_features_returns_empty_when_no_states():
    assert select_multiscale_features([], [0, 1, 2]) == []


def test_fusion_returns_stable_shape_for_token_inputs():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=16, num_scales=4)
    features = [torch.randn(3, 6, 8) for _ in range(4)]
    out = fusion(features)
    assert out.shape == (3, 16)


def test_fusion_returns_stable_shape_for_map_inputs():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=12, num_scales=4)
    features = [torch.randn(2, 8, 5, 5) for _ in range(4)]
    out = fusion(features)
    assert out.shape == (2, 12)


def test_fusion_raises_for_empty_feature_list():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=12, num_scales=4)
    with pytest.raises(ValueError):
        fusion([])


def test_fusion_pads_missing_scales_with_last_feature():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=10, num_scales=4)
    features = [torch.randn(2, 8), torch.randn(2, 8)]
    out = fusion(features)
    assert out.shape == (2, 10)


def test_fusion_uses_uniform_weights_for_non_softmax_gating():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=10, num_scales=3, gating="uniform")
    weights = fusion._weights(active_scales=3)
    assert torch.allclose(weights, torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]))


def test_fusion_backward_propagates_gradients():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=10, num_scales=4)
    features = [torch.randn(3, 8, requires_grad=True) for _ in range(4)]
    loss = fusion(features).sum()
    loss.backward()

    for projection in fusion.projections:
        assert projection.weight.grad is not None


def test_fusion_casts_half_precision_inputs_to_projection_dtype():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=10, num_scales=4)
    features = [torch.randn(2, 8, dtype=torch.float16) for _ in range(4)]
    out = fusion(features)
    assert out.dtype == fusion.projections[0].weight.dtype
    assert out.shape == (2, 10)


def test_fusion_raises_for_mismatched_input_dim():
    fusion = MultiScaleFeatureFusion(input_dim=8, output_dim=10, num_scales=4)
    bad_features = [torch.randn(2, 7) for _ in range(4)]
    with pytest.raises(RuntimeError):
        fusion(bad_features)
