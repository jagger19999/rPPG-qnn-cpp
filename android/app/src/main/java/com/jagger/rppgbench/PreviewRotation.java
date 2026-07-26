package com.jagger.rppgbench;

final class PreviewRotation {
    private PreviewRotation() {}

    static int textureTransformDegrees(int frameRotationDegrees) {
        // Camera-managed preview Surfaces arrive display-oriented on the target device.
        // The sensor angle is still used by MainActivity to size the portrait container.
        return 0;
    }
}
