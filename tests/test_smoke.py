def test_package_imports():
    import corp_default

    assert corp_default.__name__ == "corp_default"
