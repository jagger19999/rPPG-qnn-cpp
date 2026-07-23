package com.jagger.rppgbench;

public final class NativeBridge {
    static {
        System.loadLibrary("rppg_qnn_android");
    }

    private NativeBridge() {
    }

    public static native String nativeBuildIdentity();
}
