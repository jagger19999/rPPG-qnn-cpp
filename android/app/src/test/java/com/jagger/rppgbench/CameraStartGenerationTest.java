package com.jagger.rppgbench;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class CameraStartGenerationTest {
    @Test
    public void newerStartInvalidatesOlderWork() {
        CameraStartGeneration generations = new CameraStartGeneration();
        long first = generations.begin();
        long second = generations.begin();
        assertFalse(generations.isCurrent(first));
        assertTrue(generations.isCurrent(second));
    }

    @Test
    public void cancelAndDestroyInvalidateOutstandingWork() {
        CameraStartGeneration generations = new CameraStartGeneration();
        long canceled = generations.begin();
        generations.cancel();
        assertFalse(generations.isCurrent(canceled));

        long destroyed = generations.begin();
        generations.destroy();
        assertFalse(generations.isCurrent(destroyed));
        assertFalse(generations.isCurrent(generations.begin()));
    }
}
