package com.jagger.rppgbench;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public final class CameraUiSessionPolicyTest {
    @Test
    public void successfulSessionLocksUntilStopAndClearsOnlyOnAccept() {
        CameraUiSessionPolicy policy = new CameraUiSessionPolicy();

        assertUnlockedWithoutSideEffects(policy.current());

        CameraUiSessionPolicy.Decision pending = policy.begin();
        assertTrue(pending.selectorsLocked);
        assertFalse(pending.clearHistory);
        assertFalse(pending.requestFinalSnapshot);
        assertFalse(pending.retainLastResult);

        CameraUiSessionPolicy.Decision active = policy.accept();
        assertTrue(active.selectorsLocked);
        assertTrue(active.clearHistory);
        assertFalse(active.requestFinalSnapshot);
        assertFalse(active.retainLastResult);

        CameraUiSessionPolicy.Decision stopped = policy.stopRequested();
        assertFalse(stopped.selectorsLocked);
        assertFalse(stopped.clearHistory);
        assertTrue(stopped.requestFinalSnapshot);
        assertTrue(stopped.retainLastResult);
        assertUnlockedWithoutSideEffects(policy.current());
    }

    @Test
    public void failedPendingStartUnlocksWithoutClearingOrRetaining() {
        CameraUiSessionPolicy policy = new CameraUiSessionPolicy();
        policy.begin();

        assertUnlockedWithoutSideEffects(policy.fail());
        assertUnlockedWithoutSideEffects(policy.current());
    }

    @Test
    public void canceledPendingStartUnlocksWithoutClearingOrRetaining() {
        CameraUiSessionPolicy policy = new CameraUiSessionPolicy();
        policy.begin();

        assertUnlockedWithoutSideEffects(policy.stopRequested());
        assertUnlockedWithoutSideEffects(policy.current());
    }

    @Test
    public void illegalTransitionsCannotClearHistoryOrRequestSnapshots() {
        CameraUiSessionPolicy policy = new CameraUiSessionPolicy();

        assertThrows(IllegalStateException.class, policy::accept);
        assertThrows(IllegalStateException.class, policy::stop);
        policy.begin();
        assertThrows(IllegalStateException.class, policy::begin);
        assertThrows(IllegalStateException.class, policy::stop);
        policy.cancel();
        assertThrows(IllegalStateException.class, policy::fail);
    }

    private static void assertUnlockedWithoutSideEffects(
            CameraUiSessionPolicy.Decision decision) {
        assertFalse(decision.selectorsLocked);
        assertFalse(decision.clearHistory);
        assertFalse(decision.requestFinalSnapshot);
        assertFalse(decision.retainLastResult);
    }
}
