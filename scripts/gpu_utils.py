"""
gpu_utils.py - TensorFlow GPU setup helper.

- Logs available GPU devices.
- Enables memory growth to reduce OOM risk.
- Falls back to CPU cleanly if no GPUs are found.
"""

from __future__ import annotations

import logging
import os
import sys
import ctypes
from pathlib import Path
import shutil

import tensorflow as tf


def setup_gpu(log: logging.Logger | None = None) -> None:
    logger = log or logging.getLogger(__name__)
    require_gpu = os.environ.get("MGC_REQUIRE_GPU", "1") != "0"
    # Add existing shim directory early if present.
    repo_root = Path(__file__).resolve().parent.parent
    shim_dir = repo_root / "outputs" / "cuda_shim"
    if shim_dir.exists():
        try:
            os.add_dll_directory(str(shim_dir))
        except Exception:
            pass
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        logger.info(f"Added shim dir to PATH: {shim_dir}")
    # Ensure CUDA bin is on PATH (Windows) and add DLL search path.
    cuda_bins: list[Path] = []
    env_candidates = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH_V11_8"),
    ]
    for c in env_candidates:
        if c:
            cuda_bins.append(Path(c) / "bin")

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.exists():
        # Prefer v11.8 if present; otherwise pick highest version.
        candidates = sorted([p for p in cuda_root.iterdir() if p.is_dir()], reverse=True)
        preferred = None
        for p in candidates:
            if p.name.lower() == "v11.8":
                preferred = p
                break
        if preferred is None and candidates:
            preferred = candidates[0]
        if preferred:
            cuda_bins.append(preferred / "bin")

    # Sanitize PATH: drop CUDA 12 entries and move preferred CUDA bin to front.
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    cleaned_parts = []
    for p in path_parts:
        pl = p.lower()
        if "nvidia gpu computing toolkit\\cuda" in pl and ("v12" in pl or "cuda\\12" in pl):
            continue
        cleaned_parts.append(p)

    # De-duplicate and add preferred CUDA bin to front
    seen = set()
    for cuda_bin in cuda_bins:
        if not cuda_bin.exists():
            continue
        if str(cuda_bin) in seen:
            continue
        seen.add(str(cuda_bin))
        if str(cuda_bin) not in cleaned_parts:
            cleaned_parts.insert(0, str(cuda_bin))
            logger.info(f"Added CUDA bin to PATH: {cuda_bin}")
        try:
            os.add_dll_directory(str(cuda_bin))
        except Exception:
            # Not fatal; PATH still helps.
            pass
    os.environ["PATH"] = os.pathsep.join(cleaned_parts)
    logger.info("PATH head: %s", cleaned_parts[:5])
    try:
        build = tf.sysconfig.get_build_info()
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Python version: {sys.version.split()[0]}")
        logger.info(f"TensorFlow version: {tf.__version__}")
        logger.info(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
        logger.info(
            "CUDA build info: cudart=%s cudnn=%s",
            build.get("cudart_dll_name"),
            build.get("cudnn_dll_name"),
        )
        missing_cublaslt12 = False
        missing_cublaslt11 = False
        missing_cublas12 = False
        missing_cublas11 = False
        try:
            ctypes.WinDLL("cublasLt64_12.dll")
        except OSError:
            missing_cublaslt12 = True
            logger.warning("cublasLt64_12.dll not found in PATH.")
        try:
            ctypes.WinDLL("cublasLt64_11.dll")
        except OSError:
            missing_cublaslt11 = True
            logger.warning("cublasLt64_11.dll not found in PATH.")
        try:
            ctypes.WinDLL("cublas64_12.dll")
        except OSError:
            missing_cublas12 = True
            logger.warning("cublas64_12.dll not found in PATH.")
        try:
            ctypes.WinDLL("cublas64_11.dll")
        except OSError:
            missing_cublas11 = True
            logger.warning("cublas64_11.dll not found in PATH.")
    except Exception as exc:
        logger.warning(f"Could not read TF build info: {exc}")
        missing_cublaslt12 = False
        missing_cublaslt11 = False
        missing_cublas12 = False
        missing_cublas11 = False

    def _shim(name_from: str, name_to: str) -> None:
        nonlocal missing_cublaslt12, missing_cublaslt11, missing_cublas12, missing_cublas11
        repo_root = Path(__file__).resolve().parent.parent
        shim_dir = repo_root / "outputs" / "cuda_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        src_path = None
        for cuda_bin in cuda_bins:
            candidate = cuda_bin / name_from
            if candidate.exists():
                src_path = candidate
                break
        if not src_path:
            return
        shim_target = shim_dir / name_to
        try:
            shutil.copy2(src_path, shim_target)
            os.add_dll_directory(str(shim_dir))
            os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
            logger.warning("Created shim %s from %s at %s", name_to, name_from, shim_target)
            if name_to == "cublasLt64_12.dll":
                missing_cublaslt12 = False
            if name_to == "cublas64_12.dll":
                missing_cublas12 = False
        except Exception as exc:
            logger.error(f"Failed to create shim {name_to}: {exc}")

    if missing_cublaslt12 and not missing_cublaslt11:
        # Try a local shim: copy cublasLt64_11.dll to cublasLt64_12.dll.
        _shim("cublasLt64_11.dll", "cublasLt64_12.dll")

    if missing_cublas12 and not missing_cublas11:
        _shim("cublas64_11.dll", "cublas64_12.dll")

    if require_gpu and missing_cublaslt12 and missing_cublaslt11:
        raise RuntimeError(
            "GPU required, but cublasLt64_11.dll and cublasLt64_12.dll are missing. "
            "Install CUDA 11.x or add its bin directory to PATH."
        )

    if require_gpu and missing_cublas12 and missing_cublas11:
        raise RuntimeError(
            "GPU required, but cublas64_11.dll and cublas64_12.dll are missing. "
            "Install CUDA 11.x or add its bin directory to PATH."
        )
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("No GPU detected — running on CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:  # pragma: no cover - TF backend specific
            logger.warning(f"Could not set memory growth for {gpu}: {exc}")

    logger.info(f"GPUs detected: {[g.name for g in gpus]}")
