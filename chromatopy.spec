# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src/chromatopy/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
    ('src/chromatopy/config/integration_settings.json', 'src/chromatopy/config'),
    ('src/chromatopy/config/chromatopy_gdgt_config.json', 'src/chromatopy/config'),],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='chromatopy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='chromatopy',
)
app = BUNDLE(
    coll,
    name='chromatopy.app',
    icon=None,
    bundle_identifier=None,
)
