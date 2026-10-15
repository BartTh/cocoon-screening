# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build.

Replaces the long command line in .github/workflows/build-macos.yml so that what the
bundle contains is reviewable, and so the version can be set in Info.plist directly -
editing the plist after the build invalidated PyInstaller's ad-hoc signature and macOS
then refused to launch the app ("damaged"), which is what broke v0.3.0.

EXCLUDES is the whole point: the app draws no plots, has no web view and runs no QML, yet
all of that was being shipped. Anything removed here must be absent from the running app,
so add with care and test the build.
"""

from pathlib import Path

VERSION_FILE = Path("version.txt")
VERSION = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "dev"

EXCLUDES = [
    # Plotting and notebooks: nothing in the app draws a chart.
    "matplotlib", "IPython", "notebook", "jupyter", "jupyter_core", "nbformat",
    # StarDist runs through onnxruntime, with its post-processing ported to
    # processing/stardist_post.py. None of these are app dependencies; keep them out even
    # if a stray install (e.g. for the port tests) puts them in the environment.
    "tensorflow", "tensorboard", "tensorboard_data_server", "tensorflow_estimator",
    "keras", "h5py", "google.protobuf.internal._api_implementation",
    "stardist", "csbdeep", "skimage", "scipy", "numba", "llvmlite", "tifffile",
    # Test and build tooling that sneaks in through dependencies.
    "pytest", "_pytest", "setuptools._distutils", "pip", "sphinx",
    # Other GUI toolkits: the app is PySide6 only.
    "tkinter", "PyQt5", "PyQt6", "PIL.ImageTk", "PIL.ImageQt",
    # Qt modules this app never touches. QtWebEngine alone is well over 100 MB.
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2", "PySide6.QtVirtualKeyboard",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtSerialPort", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools", "PySide6.QtBluetooth",
    "PySide6.QtNfc", "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSensors", "PySide6.QtSpatialAudio", "PySide6.QtTextToSpeech",
]

datas = [
    ("configs", "configs"),
    ("models/bee_cocoon_stardist/config.json", "models/bee_cocoon_stardist"),
    ("models/bee_cocoon_stardist/model.onnx", "models/bee_cocoon_stardist"),
]
if VERSION_FILE.exists():
    datas.append(("version.txt", "."))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Cocoon Annotation Tool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                 # UPX breaks code signing on macOS
    console=False,             # --windowed
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Cocoon Annotation Tool",
)

app = BUNDLE(
    coll,
    name="Cocoon Annotation Tool.app",
    icon=None,
    bundle_identifier="ch.thomsonimaging.cocoon-annotation-tool",
    version=VERSION.lstrip("v"),       # Info.plist, written before signing
    info_plist={
        "CFBundleShortVersionString": VERSION.lstrip("v"),
        "CFBundleVersion": VERSION.lstrip("v"),
        "NSHighResolutionCapable": True,
    },
)
