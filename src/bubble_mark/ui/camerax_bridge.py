"""CameraX bridge for Android.

On Android (Briefcase build), this module sets up a CameraX ``ImageAnalysis``
use-case and delivers each frame as an RGB NumPy array to a Python callback.

On all other platforms the public API is a no-op so the rest of the
application can import this module unconditionally.

Usage::

    from bubble_mark.ui.camerax_bridge import start_camera, stop_camera

    def on_frame(rgb: np.ndarray) -> None:
        ...  # called on a CameraX worker thread

    start_camera(on_frame)
    ...
    stop_camera()
"""

from __future__ import annotations

import sys
from typing import Callable

import numpy as np

# The callback type: receives an HxWx3 uint8 RGB array.
FrameCallback = Callable[[np.ndarray], None]

# Module-level reference so stop_camera() can unbind.
_provider = None


def _is_android() -> bool:
    return sys.platform == "android" or "ANDROID_DATA" in __import__("os").environ


# ---------------------------------------------------------------------------
# Android implementation
# ---------------------------------------------------------------------------


def _start_android(callback: FrameCallback) -> None:
    global _provider
    # Idempotency guard: silently skip if the camera is already running.
    if _provider is not None:
        return

    from jnius import PythonJavaClass, autoclass, java_method  # type: ignore[import]

    ProcessCameraProvider = autoclass("androidx.camera.lifecycle.ProcessCameraProvider")
    ImageAnalysis = autoclass("androidx.camera.core.ImageAnalysis")
    CameraSelector = autoclass("androidx.camera.core.CameraSelector")
    ContextCompat = autoclass("androidx.core.content.ContextCompat")
    Executors = autoclass("java.util.concurrent.Executors")
    # Briefcase's MainActivity is the lifecycle owner.
    MainActivity = autoclass("org.beeware.android.MainActivity")

    activity = MainActivity.sActivity
    # Main executor: used only for bindToLifecycle (must run on UI thread).
    main_executor = ContextCompat.getMainExecutor(activity)
    # Background executor: used for frame analysis to avoid blocking the UI thread.
    analysis_executor = Executors.newSingleThreadExecutor()

    class _Analyzer(PythonJavaClass):
        """Python implementation of ImageAnalysis.Analyzer."""

        __javainterfaces__ = ["androidx/camera/core/ImageAnalysis$Analyzer"]

        # Store callback as a class attribute so jnius can serialise the instance.
        _cb: FrameCallback | None = None

        @java_method("(Landroidx/camera/core/ImageProxy;)V")
        def analyze(self, image_proxy) -> None:  # noqa: D102
            try:
                planes = image_proxy.getPlanes()
                buf = planes[0].getBuffer()
                width = image_proxy.getWidth()
                height = image_proxy.getHeight()
                # RGBA_8888: single plane, row-stride may be > width*4
                row_stride = planes[0].getRowStride()
                raw = np.frombuffer(bytes(buf), dtype=np.uint8)
                if row_stride == width * 4:
                    rgba = raw.reshape((height, width, 4))
                else:
                    rgba = raw.reshape((height, row_stride // 4, 4))[:, :width, :]
                rgb = rgba[:, :, :3].copy()
                if self._cb is not None:
                    self._cb(rgb)
            finally:
                image_proxy.close()

    analyzer = _Analyzer()
    analyzer._cb = callback

    analysis = (
        ImageAnalysis.Builder()
        # RGBA_8888 = 2 (single plane, trivial numpy reshape)
        .setOutputImageFormat(2)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()
    )
    # Use a background executor so numpy/OpenCV work never runs on the UI thread.
    analysis.setAnalyzer(analysis_executor, analyzer)

    future = ProcessCameraProvider.getInstance(activity)

    class _ProviderCallback(PythonJavaClass):
        __javainterfaces__ = ["java/lang/Runnable"]

        @java_method("()V")
        def run(self) -> None:
            global _provider
            try:
                provider = future.get()
                _provider = provider
                provider.bindToLifecycle(
                    activity,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    analysis,
                )
            except Exception as exc:
                import logging

                logging.getLogger(__name__).error("CameraX bind failed: %s", exc)

    # bindToLifecycle must be called on the main thread.
    future.addListener(_ProviderCallback(), main_executor)


def _stop_android() -> None:
    global _provider
    if _provider is not None:
        try:
            _provider.unbindAll()
        except Exception:
            pass
        _provider = None


# ---------------------------------------------------------------------------
# Desktop stub
# ---------------------------------------------------------------------------


def _start_stub(callback: FrameCallback) -> None:  # noqa: D401
    """No-op on non-Android platforms."""


def _stop_stub() -> None:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

if _is_android():
    start_camera = _start_android
    stop_camera = _stop_android
else:
    start_camera = _start_stub
    stop_camera = _stop_stub
