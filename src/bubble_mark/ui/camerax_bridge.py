"""Camera bridge for Android and iOS.

On Android (Briefcase build), this module sets up a CameraX ``ImageAnalysis``
use-case and delivers each frame as an RGB NumPy array to a Python callback.

On iOS (Briefcase build), this module sets up an ``AVCaptureSession`` with an
``AVCaptureVideoDataOutput`` using AVFoundation via rubicon-objc, and delivers
each frame as an RGB NumPy array to the same callback interface.

On all other platforms the public API is a no-op so the rest of the
application can import this module unconditionally.

Usage::

    from bubble_mark.ui.camerax_bridge import start_camera, stop_camera

    def on_frame(rgb: np.ndarray) -> None:
        ...  # called on a camera worker thread

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

# Module-level references so stop_camera() can unbind / release resources.
_provider = None       # Android: ProcessCameraProvider
_ios_session = None    # iOS: AVCaptureSession
_ios_delegate = None   # iOS: _FrameDelegate (kept to prevent GC)


def _is_android() -> bool:
    return sys.platform == "android" or "ANDROID_DATA" in __import__("os").environ


def _is_ios() -> bool:
    return sys.platform == "ios"


# ---------------------------------------------------------------------------
# Android implementation
# ---------------------------------------------------------------------------


def _start_android(callback: FrameCallback) -> bool:
    """Start CameraX on Android.

    Returns ``True`` when CAMERA permission is granted and CameraX startup has
    been initiated (or the camera was already running); note that the binding
    itself is asynchronous and may still fail.  Returns ``False`` when the
    CAMERA permission has just been requested from the user (the caller should
    prompt the user to grant it and tap "Open Camera" again).
    """
    global _provider
    # Idempotency guard: silently skip if the camera is already running.
    if _provider is not None:
        return True

    from jnius import PythonJavaClass, autoclass, java_method  # type: ignore[import]

    MainActivity = autoclass("org.beeware.android.MainActivity")
    activity = MainActivity.sActivity

    # ------------------------------------------------------------------ #
    # Runtime CAMERA permission check (required on Android 6+)           #
    # ------------------------------------------------------------------ #
    ContextCompat = autoclass("androidx.core.content.ContextCompat")
    ActivityCompat = autoclass("androidx.core.app.ActivityCompat")
    PackageManager = autoclass("android.content.pm.PackageManager")
    _CAMERA_PERMISSION = "android.permission.CAMERA"
    _PERMISSION_GRANTED = PackageManager.PERMISSION_GRANTED
    _CAMERA_PERMISSION_REQUEST_CODE = 1

    if (
        ContextCompat.checkSelfPermission(activity, _CAMERA_PERMISSION)
        != _PERMISSION_GRANTED
    ):
        ActivityCompat.requestPermissions(
            activity, [_CAMERA_PERMISSION], _CAMERA_PERMISSION_REQUEST_CODE
        )
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
# iOS implementation (AVFoundation via rubicon-objc)
# ---------------------------------------------------------------------------


def _start_ios(callback: FrameCallback) -> bool:
    """Start AVFoundation camera capture on iOS.

    Returns ``True`` when an ``AVCaptureSession`` has been started (or was
    already running).  Returns ``False`` when camera permission is not yet
    granted (the caller should prompt the user to grant it in Settings and
    tap "Open Camera" again) or when the capture device is unavailable.
    """
    global _ios_session, _ios_delegate

    # Idempotency guard: silently skip if the session is already running.
    if _ios_session is not None:
        return True

    import ctypes
    import logging

    log = logging.getLogger(__name__)

    from rubicon.objc import ObjCClass, objc_method  # type: ignore[import]
    from rubicon.objc.api import NSObject  # type: ignore[import]

    # ── CoreMedia / CoreVideo C-function bindings ────────────────────────
    try:
        CoreMedia = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreMedia.framework/CoreMedia"
        )
        CoreVideo = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/CoreVideo.framework/CoreVideo"
        )
    except OSError as exc:
        log.error("Could not load iOS system frameworks: %s", exc)
        return False

    CoreMedia.CMSampleBufferGetImageBuffer.restype = ctypes.c_void_p
    CoreMedia.CMSampleBufferGetImageBuffer.argtypes = [ctypes.c_void_p]

    CoreVideo.CVPixelBufferLockBaseAddress.restype = ctypes.c_int32
    CoreVideo.CVPixelBufferLockBaseAddress.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    CoreVideo.CVPixelBufferUnlockBaseAddress.restype = ctypes.c_int32
    CoreVideo.CVPixelBufferUnlockBaseAddress.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32
    ]
    CoreVideo.CVPixelBufferGetBaseAddress.restype = ctypes.c_void_p
    CoreVideo.CVPixelBufferGetBaseAddress.argtypes = [ctypes.c_void_p]
    CoreVideo.CVPixelBufferGetWidth.restype = ctypes.c_size_t
    CoreVideo.CVPixelBufferGetWidth.argtypes = [ctypes.c_void_p]
    CoreVideo.CVPixelBufferGetHeight.restype = ctypes.c_size_t
    CoreVideo.CVPixelBufferGetHeight.argtypes = [ctypes.c_void_p]
    CoreVideo.CVPixelBufferGetBytesPerRow.restype = ctypes.c_size_t
    CoreVideo.CVPixelBufferGetBytesPerRow.argtypes = [ctypes.c_void_p]

    # kCVPixelFormatType_32BGRA = 'BGRA' four-char code (0x42475241)
    _kCVPixelFormatType_32BGRA = 1111970369
    # kCVPixelBufferLock_ReadOnly = 0x00000001
    _kCVPixelBufferLock_ReadOnly = 1

    # ── AVFoundation classes ─────────────────────────────────────────────
    AVCaptureDevice = ObjCClass("AVCaptureDevice")
    AVCaptureDeviceInput = ObjCClass("AVCaptureDeviceInput")
    AVCaptureVideoDataOutput = ObjCClass("AVCaptureVideoDataOutput")
    AVCaptureSession = ObjCClass("AVCaptureSession")
    NSNumber = ObjCClass("NSNumber")
    NSMutableDictionary = ObjCClass("NSMutableDictionary")

    # AVMediaTypeVideo = @"vide" (QuickTime four-character code for video)
    _AVMediaTypeVideo = "vide"

    # ── Runtime camera permission check ──────────────────────────────────
    # authorizationStatusForMediaType: 0=NotDetermined 1=Restricted
    #                                   2=Denied       3=Authorized
    status = AVCaptureDevice.authorizationStatusForMediaType_(_AVMediaTypeVideo)

    if status == 2:  # Denied
        log.warning("Camera permission denied; grant it in iOS Settings.")
        return False

    if status == 0:  # NotDetermined — trigger the system permission dialog
        # requestAccessForMediaType:completionHandler: shows the system alert.
        # The user must tap "Open Camera" again after granting.
        def _on_permission(granted: bool) -> None:
            if granted:
                log.info("Camera permission granted; tap 'Open Camera' to continue.")
            else:
                log.warning("Camera permission denied; grant it in iOS Settings.")

        AVCaptureDevice.requestAccessForMediaType_completionHandler_(
            _AVMediaTypeVideo, _on_permission
        )
        return False

    # status == 3 (Authorized) or 1 (Restricted — attempt anyway)

    # ── Capture device ───────────────────────────────────────────────────
    device = AVCaptureDevice.defaultDeviceWithMediaType_(_AVMediaTypeVideo)
    if not device:
        log.error("No AVCaptureDevice found for video.")
        return False

    device_input = AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if not device_input:
        log.error("Could not create AVCaptureDeviceInput.")
        return False

    # ── Frame-delivery delegate ───────────────────────────────────────────
    # We define the class inside the function so it closes over the ctypes
    # bindings and the callback without relying on module-level state.

    class _FrameDelegate(NSObject):
        """AVCaptureVideoDataOutputSampleBufferDelegate implementation."""

        _cb: FrameCallback | None = None

        @objc_method
        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, output, sampleBuffer, connection
        ) -> None:
            try:
                pb = CoreMedia.CMSampleBufferGetImageBuffer(
                    sampleBuffer.ptr.value
                )
                if not pb:
                    return
                CoreVideo.CVPixelBufferLockBaseAddress(
                    pb, _kCVPixelBufferLock_ReadOnly
                )
                try:
                    base = CoreVideo.CVPixelBufferGetBaseAddress(pb)
                    w = CoreVideo.CVPixelBufferGetWidth(pb)
                    h = CoreVideo.CVPixelBufferGetHeight(pb)
                    bpr = CoreVideo.CVPixelBufferGetBytesPerRow(pb)
                    raw = (ctypes.c_uint8 * (bpr * h)).from_address(base)
                    # BGRA pixel layout → strip alpha, crop row padding → RGB
                    bgra = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (h, bpr // 4, 4)
                    )
                    rgb = bgra[:, :w, 2::-1].copy()
                    if self._cb is not None:
                        self._cb(rgb)
                finally:
                    CoreVideo.CVPixelBufferUnlockBaseAddress(
                        pb, _kCVPixelBufferLock_ReadOnly
                    )
            except Exception as exc:
                log.error("AVFoundation frame error: %s", exc)

    delegate = _FrameDelegate.alloc().init()
    delegate._cb = callback

    # ── Video data output ─────────────────────────────────────────────────
    video_settings = NSMutableDictionary.alloc().init()
    video_settings.setObject_forKey_(
        NSNumber.numberWithUnsignedInt_(_kCVPixelFormatType_32BGRA),
        "CVPixelBufferPixelFormatTypeKey",
    )

    output = AVCaptureVideoDataOutput.alloc().init()
    output.videoSettings = video_settings
    output.alwaysDiscardsLateVideoFrames = True

    # Serial dispatch queue for frame delivery (runs off the main thread)
    libdispatch = ctypes.cdll.LoadLibrary("libdispatch.dylib")
    libdispatch.dispatch_queue_create.restype = ctypes.c_void_p
    libdispatch.dispatch_queue_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
    queue_ptr = libdispatch.dispatch_queue_create(
        b"com.wdconinc.bubble_mark.camera", None
    )

    from rubicon.objc.api import ObjCInstance  # type: ignore[import]

    output.setSampleBufferDelegate_queue_(delegate, ObjCInstance(queue_ptr))

    # ── Session ───────────────────────────────────────────────────────────
    session = AVCaptureSession.alloc().init()
    if session.canAddInput_(device_input):
        session.addInput_(device_input)
    if session.canAddOutput_(output):
        session.addOutput_(output)

    session.startRunning()

    _ios_session = session
    _ios_delegate = delegate  # prevent GC of the delegate
    return True


def _stop_ios() -> None:
    global _ios_session, _ios_delegate
    if _ios_session is not None:
        try:
            _ios_session.stopRunning()
        except Exception:
            pass
        _ios_session = None
        _ios_delegate = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

if _is_android():
    start_camera = _start_android
    stop_camera = _stop_android
elif _is_ios():
    start_camera = _start_ios
    stop_camera = _stop_ios
else:
    start_camera = _start_stub
    stop_camera = _stop_stub
