package com.jagger.rppgbench.watch;

import java.util.List;
import java.util.function.Consumer;

public interface WatchBleBackend {
    interface DeviceHandle {
        String id();

        String name();
    }

    List<DeviceHandle> scan(double timeoutSec) throws Exception;

    void connect(
            DeviceHandle handle,
            Consumer<byte[]> onNotification,
            Runnable onDisconnect) throws Exception;

    void disconnect() throws Exception;
}
