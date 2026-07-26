package com.jagger.rppgbench;

final class CameraStartGeneration {
    private long generation;
    private boolean destroyed;

    synchronized long begin() {
        return ++generation;
    }

    synchronized void cancel() {
        ++generation;
    }

    synchronized void destroy() {
        destroyed = true;
        ++generation;
    }

    synchronized boolean isCurrent(long candidate) {
        return !destroyed && candidate == generation;
    }
}
