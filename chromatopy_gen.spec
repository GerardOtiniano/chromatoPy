import os
import matplotlib
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE

# 1) Gather any hidden imports PyInstaller might miss
hidden_imports = (
    collect_submodules('numpy')
  + collect_submodules('scipy')
  + [
      'matplotlib.backends.backend_macosx',  # native macOS GUI
      'matplotlib.backends.backend_tkagg',   # fallback TkAgg
      'tkinter',                             # for TkAgg
    ]
)

# 2) Collect Matplotlib data files (style sheets, fonts, etc.)
datas = []
mpl_data = matplotlib.get_data_path()
for root, dirs, files in os.walk(mpl_data):
    for fn in files:
        src = os.path.join(root, fn)
        dst = os.path.join('mpl-data', os.path.relpath(src, mpl_data))
        datas.append((src, dst))

# 3) Include your ChromatoPy config JSON
datas.append((
    os.path.join('src', 'chromatopy', 'config', 'integration_settings.json'),
    os.path.join('chromatopy', 'config')
))

datas.append((
    os.path.join('src', 'chromatopy', 'config', 'plot_settings.json'),
    os.path.join('chromatopy', 'config')
))

icon_src_dir = os.path.join('src', 'chromatopy', 'Icons')
for icon_file in os.listdir(icon_src_dir):
    if icon_file.endswith('.png'):
        datas.append((
            os.path.join(icon_src_dir, icon_file),
            os.path.join('chromatopy', 'Icons')
        ))

# 4) Build Analysis
a = Analysis(
    ['src/chromatopy/__main__.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# 5) Create the Python archive
pyz = PYZ(a.pure, a.zipped_data)

# 6) Build the executable
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='chromatopy_gen',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,         # switch to True if you need a terminal window
    icon='AppIcon/chromatopy_icon.icns',
)

# 7) Collect into a folder
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name='chromatopy_gen',
)

# 8) Bundle as a macOS .app
app = BUNDLE(
    coll,
    name='chromatopy.app',
    icon='AppIcon/chromatopy_icon.icns',
    bundle_identifier='com.otiniano.chromatopygen',
    info_plist={
        'CFBundleName':               'chromatopy_gen',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion':            '1',
        'LSMinimumSystemVersion':     '10.12',
    },
)
