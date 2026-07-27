from src.router.sam3_output_utils import normalize_sam3_results, sam3_error_result


class _TensorLike:
    def __init__(self, shape):
        self.shape = shape


def test_normalize_sam3_results_keeps_non_empty_tensors():
    results = {
        'masks': _TensorLike((1, 32, 32)),
        'boxes': _TensorLike((1, 4)),
        'scores': _TensorLike((1,)),
    }

    normalized = normalize_sam3_results(results, empty_tensor_factory=lambda: _TensorLike((0,)))

    assert normalized['masks'] is results['masks']
    assert normalized['boxes'] is results['boxes']
    assert normalized['scores'] is results['scores']


def test_normalize_sam3_results_returns_empty_lists_for_empty_shapes():
    results = {
        'masks': _TensorLike(()),
        'boxes': _TensorLike(()),
        'scores': _TensorLike(()),
    }

    normalized = normalize_sam3_results(results, empty_tensor_factory=lambda: _TensorLike((0,)))

    assert normalized == {'masks': [], 'boxes': [], 'scores': []}


def test_sam3_error_result_contains_error_message():
    err = RuntimeError('boom')
    payload = sam3_error_result(err)
    assert payload == {'masks': [], 'boxes': [], 'scores': [], 'error': 'boom'}
