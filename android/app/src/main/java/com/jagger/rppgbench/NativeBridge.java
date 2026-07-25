package com.jagger.rppgbench;

public final class NativeBridge {
    static {
        System.loadLibrary("rppg_qnn_android");
    }

    private NativeBridge() {
    }

    public static native String nativeBuildIdentity();
    public static native String nativeListCameras();
    public static native long nativeCreate(String cameraId, int width, int height, int fps);
    public static native String nativeConfigureProcessing(
            long handle, String method, String cascadePath, String outputDirectory,
            boolean deepEnabled, String modelPath);
    public static native void nativeSetPreviewSurface(long handle, android.view.Surface surface);
    public static native String nativeStart(long handle);
    public static native String nativeStop(long handle);
    public static native void nativeDestroy(long handle);
    public static native String nativeGetStatus(long handle);
    public static native byte[] nativeGetRoiJpeg(long handle);
}
