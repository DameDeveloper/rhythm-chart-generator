# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: build a single-file Rhythm Chart Generator .exe.

Run from the backend/ directory:

    python -m PyInstaller rhythm_chart_generator.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# Bundle the built frontend and the pattern definitions next to the modules so
# their `Path(__file__).parent / "..."` lookups resolve inside the bundle.
datas += [
    ("../frontend/dist", "dist"),
    ("patterns", "patterns"),
]

# Third-party packages that ship data files / binaries or use dynamic imports
# PyInstaller cannot follow on its own.
for pkg in ("imageio_ffmpeg", "yt_dlp", "uvicorn"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules("uvicorn")
hiddenimports += [
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "anyio._backends._asyncio",
]


a = Analysis(
    ["desktop_app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    # This app only needs numpy + web stack + yt-dlp/ffmpeg.  The build env has
    # heavy ML packages installed that PyInstaller would otherwise pull in via
    # optional import chains, ballooning the exe to ~1 GB.  Exclude them.
    excludes=[
        "tkinter",
        "torch", "torchvision", "torchaudio",
        "transformers", "datasets", "huggingface_hub", "tokenizers", "safetensors",
        "accelerate", "bitsandbytes", "peft", "sentencepiece",
        "scipy", "sklearn", "pandas", "matplotlib", "sympy", "numba", "llvmlite",
        "cv2", "PIL", "skimage", "joblib", "networkx",
        "IPython", "jedi", "notebook", "ipykernel", "jupyter_client",
        "tensorflow", "keras", "onnx", "onnxruntime", "triton",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="RhythmChartGenerator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="../assets/app.ico",
    version="version_info.txt",
)
