package com.jagger.rppgbench;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public final class PreviewRotationTest {
    @Test
    public void textureTransformDoesNotRotateCameraManagedSurface() {
        assertEquals(0, PreviewRotation.textureTransformDegrees(90));
        assertEquals(0, PreviewRotation.textureTransformDegrees(270));
        assertEquals(0, PreviewRotation.textureTransformDegrees(0));
    }
}
