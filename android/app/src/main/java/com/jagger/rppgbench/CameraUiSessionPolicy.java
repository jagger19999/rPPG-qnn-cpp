package com.jagger.rppgbench;

/** Pure state policy for camera-session UI controls and result-history side effects. */
public final class CameraUiSessionPolicy {
    private enum State {
        IDLE,
        PENDING,
        ACTIVE
    }

    public static final class Decision {
        public final boolean selectorsLocked;
        public final boolean clearHistory;
        public final boolean requestFinalSnapshot;
        public final boolean retainLastResult;

        private Decision(
                boolean selectorsLocked,
                boolean clearHistory,
                boolean requestFinalSnapshot,
                boolean retainLastResult) {
            this.selectorsLocked = selectorsLocked;
            this.clearHistory = clearHistory;
            this.requestFinalSnapshot = requestFinalSnapshot;
            this.retainLastResult = retainLastResult;
        }
    }

    private State state = State.IDLE;

    public Decision current() {
        return new Decision(state != State.IDLE, false, false, false);
    }

    public Decision begin() {
        require(State.IDLE, "begin");
        state = State.PENDING;
        return new Decision(true, false, false, false);
    }

    public Decision accept() {
        require(State.PENDING, "accept");
        state = State.ACTIVE;
        return new Decision(true, true, false, false);
    }

    public Decision stop() {
        require(State.ACTIVE, "stop");
        state = State.IDLE;
        return new Decision(false, false, true, true);
    }

    public Decision fail() {
        require(State.PENDING, "fail");
        state = State.IDLE;
        return new Decision(false, false, false, false);
    }

    public Decision cancel() {
        require(State.PENDING, "cancel");
        state = State.IDLE;
        return new Decision(false, false, false, false);
    }

    /** Ends an active session or cancels a pending one; idle requests are harmless. */
    public Decision stopRequested() {
        switch (state) {
            case ACTIVE:
                return stop();
            case PENDING:
                return cancel();
            case IDLE:
                return current();
            default:
                throw new IllegalStateException("unknown camera UI session state");
        }
    }

    private void require(State expected, String transition) {
        if (state != expected) {
            throw new IllegalStateException(
                    "cannot " + transition + " camera UI session from " + state);
        }
    }
}
