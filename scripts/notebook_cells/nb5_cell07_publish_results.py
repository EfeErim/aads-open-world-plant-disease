# Auto-extracted from colab_notebooks/5_calibrate_router_handoff_thresholds.ipynb cell 7.
# Keep notebook execute-only cells thin; edit behavior here.

if PUBLISH_RESULTS_TO_GIT:
    import shutil
    import subprocess
    from datetime import datetime, timezone

    publish_root = Path(PUBLISH_RESULTS_ROOT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target_dir = publish_root / stamp
    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for path_value in (BASELINE_EVAL_OUTPUT, CALIBRATION_OUTPUT, HOLDOUT_VALIDATION_OUTPUT):
        source_path = Path(path_value)
        if source_path.exists():
            destination = target_dir / source_path.name
            shutil.copy2(source_path, destination)
            copied.append(str(destination))

    # Also include failure analysis and selection summary if present
    dev_result = (globals().get('calibration_result') or {})
    if dev_result:
        if dev_result.get('failure_analysis'):
            (target_dir / 'dev_failure_analysis.json').write_text(
                json.dumps(dev_result.get('failure_analysis'), indent=2), encoding='utf-8'
            )
            copied.append(str(target_dir / 'dev_failure_analysis.json'))
        # archive eligible/rejected lists
        (target_dir / 'dev_selection_summary.json').write_text(
            json.dumps(dev_result.get('selection_summary') or {}, indent=2), encoding='utf-8'
        )
        copied.append(str(target_dir / 'dev_selection_summary.json'))

    summary = {
        'created_at': stamp,
        'copied': copied,
        'calibration_recommended': (globals().get('calibration_result') or {}).get('recommended', {}),
        'holdout_recommended': (globals().get('holdout_result') or {}).get('recommended', {}),
    }
    (target_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    copied.append(str(target_dir / 'summary.json'))
    print('[PUBLISH] copied result artifacts')
    print(json.dumps(copied, indent=2))

    if AUTO_COMMIT_PUSH_RESULTS:
        sparse_add = subprocess.run(
            ['git', 'sparse-checkout', 'add', 'runs/_index/router_calibration'],
            check=False,
            text=True,
            capture_output=True,
        )
        if sparse_add.returncode not in (0, 128):
            print(f'[PUBLISH] sparse-checkout add returncode={sparse_add.returncode}')
            if sparse_add.stderr:
                print(sparse_add.stderr)
        add = subprocess.run(
            ['git', 'add', '-f', str(target_dir)],
            check=False,
            text=True,
            capture_output=True,
        )
        if add.returncode != 0:
            print(f'[PUBLISH] git add failed returncode={add.returncode}')
            if add.stdout:
                print(add.stdout)
            if add.stderr:
                print(add.stderr)
            raise RuntimeError('Notebook 5 publish failed while staging router calibration results.')
        commit_message = f'Add router calibration results {stamp}'
        commit = subprocess.run(['git', 'commit', '-m', commit_message], check=False)
        if commit.returncode == 0:
            push = subprocess.run(['git', 'push', 'origin', 'master'], check=False)
            print(f'[PUBLISH] git push returncode={push.returncode}')
        else:
            print(f'[PUBLISH] git commit skipped/failed returncode={commit.returncode}')
else:
    print('[PUBLISH] skipped')
