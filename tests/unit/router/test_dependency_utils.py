from src.router.dependency_utils import check_vlm_dependencies


def test_check_vlm_dependencies_reports_old_transformers_and_missing_optional():
    def fake_import(name):
        if name == 'transformers':
            class _T:
                __version__ = '4.40.0'
            return _T
        raise ImportError(name)

    diagnostics = check_vlm_dependencies(import_module=fake_import)

    assert "outside the repo's validated range" in diagnostics['transformers_warning']
    assert '5.1.0' in diagnostics['transformers_warning']
    assert diagnostics['missing_deps'] == ['open-clip-torch']
    assert diagnostics['install_command'] == '!pip install open-clip-torch'


def test_check_vlm_dependencies_ok_path():
    def fake_import(name):
        if name == 'transformers':
            class _T:
                __version__ = '5.1.0'
            return _T
        if name == 'open_clip':
            return object()
        raise ImportError(name)

    diagnostics = check_vlm_dependencies(import_module=fake_import)

    assert diagnostics['transformers_warning'] is None
    assert diagnostics['missing_deps'] == []
    assert diagnostics['install_command'] is None


def test_check_vlm_dependencies_reports_newer_minor_outside_validated_range():
    def fake_import(name):
        if name == 'transformers':
            class _T:
                __version__ = '5.2.0'
            return _T
        if name == 'open_clip':
            return object()
        raise ImportError(name)

    diagnostics = check_vlm_dependencies(import_module=fake_import)

    assert "outside the repo's validated range" in diagnostics['transformers_warning']
    assert '5.1.0' in diagnostics['transformers_warning']


def test_check_vlm_dependencies_reports_future_major_version():
    def fake_import(name):
        if name == 'transformers':
            class _T:
                __version__ = '6.0.0'
            return _T
        if name == 'open_clip':
            return object()
        raise ImportError(name)

    diagnostics = check_vlm_dependencies(import_module=fake_import)

    assert "outside the repo's validated range" in diagnostics['transformers_warning']
    assert '5.1.0' in diagnostics['transformers_warning']
