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

def _start_android(callback: FrameCallback) -> bool:
    """Start CameraX on Android.

    Returns ``True`` when the camera pipeline has been successfully started (or
    was already running), and ``False`` when the CAMERA permission has just been
    requested from the user (the caller should prompt the user to grant it and
    tap "Open Camera" again).
    """
    global _provider
    # Idempotency guard: silently skip if the camera is already running.
    if _provider is not None:
        return True

    from jnius import autoclass, PythonJavaClass, java_method  # type: ignore[import]

    MainActivity = autoclass("org.beeware.android.MainActivity")
    activity = MainActivity.sActivity

    # ------------------------------------------------------------------ #
    # Runtime CAMERA permission check (required on Android 6+)           #
    # ------------------------------------------------------------------ #
    ContextCompat = autoclass("androidx.core.content.ContextCompat")
    ActivityCompat = autoclass("androidx.core.app.ActivityCompat")
    _CAMERA_PERMISSION = "android.permission.CAMERA"
    _PERMISSION_GRANTED = 0  # PackageManager.PERMISSION_GRANTED
    _CAMERA_PERMISSION_REQUEST_CODE = 1

    if ContextCompat.checkSelfPermission(activity, _CAMERA_PERMISSION) != _PERMISSION_GRANTED:
        ActivityCompat.requestPermissions(activity, [_CAMERA_PERMISSION], _CAMERA_PERMISSION_REQUEST_CODE)
        return False

    ProcessCameraProvider = autoclass("androidx.camera.lifecycle.ProcessCameraProvider")
    ImageAnalysis = autoclass("androidx.camera.core.ImageAnalysis")
    CameraSelector = autoclass("androidx.camera.core.CameraSelector")
    Executors = autoclass("java.util.concurrent.Executors")
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
        .setBackpressureStrategy(
            ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
        )
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
    return True


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

def _start_stub(callback: FrameCallback) -> bool:  # noqa: D401
    """No-op on non-Android platforms."""
    return True


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
