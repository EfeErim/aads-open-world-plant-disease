# Auto-extracted from colab_notebooks/3_validate_exported_adapter_directly.ipynb cell 7.
# Keep notebook execute-only cells thin; edit behavior here.

if not BATCH_IMAGE_DIR:
    print('Opsiyonel klasor testi icin BATCH_IMAGE_DIR degerini bir goruntu klasorune ayarlayin.')
else:
    rows = predict_image_folder(
        BATCH_IMAGE_DIR,
        CROP_NAME,
        adapter_dir=ADAPTER_DIR,
        adapter_root=ADAPTER_ROOT,
        config_env=CONFIG_ENV,
        device=DEVICE,
    )
    if pd is not None:
        df = pd.DataFrame(rows)
        display(df)
    else:
        print(rows)

    predicted_counts = Counter(row['predicted_class'] for row in rows if row.get('predicted_class'))
    ood_count = sum(1 for row in rows if row.get('is_ood') is True)
    error_count = sum(1 for row in rows if row.get('error'))

    print('predicted_class_counts:', dict(predicted_counts))
    print('ood_count:', ood_count)
    print('error_count:', error_count)
