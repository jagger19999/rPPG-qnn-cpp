package com.jagger.rppgbench.watch;

import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

public final class WatchSampleStoreTest {
    @Test
    public void acceptMeasurementSetsStreamingAndLatestBpm() {
        WatchSampleStore store = new WatchSampleStore();
        store.setStatus(WatchContracts.WatchConnectionStatus.CONNECTING);
        store.acceptMeasurement(10.0, 72, new double[0], "aa:bb", "GT 5 Pro");

        WatchContracts.WatchHeartRateSnapshot snapshot = store.snapshot(10.1);
        assertEquals(WatchContracts.WatchConnectionStatus.STREAMING, snapshot.status);
        assertEquals(72, snapshot.latestSample.bpm);
        assertEquals(1, snapshot.samples.size());
    }

    @Test
    public void historyIsBoundedTo180Seconds() {
        WatchSampleStore store = new WatchSampleStore();
        store.setStatus(WatchContracts.WatchConnectionStatus.STREAMING);
        store.acceptMeasurement(1.0, 60, new double[0], "aa", "watch");
        store.acceptMeasurement(50.0, 61, new double[0], "aa", "watch");
        store.acceptMeasurement(200.0, 62, new double[0], "aa", "watch");

        WatchContracts.WatchHeartRateSnapshot snapshot = store.snapshot(200.0);
        assertEquals(2, snapshot.samples.size());
        assertEquals(62, snapshot.latestSample.bpm);
        assertEquals(50.0, snapshot.samples.get(0).receivedMonotonicSec, 1e-9);
    }

    @Test
    public void snapshotMarksStaleAfterTwoSeconds() {
        WatchSampleStore store = new WatchSampleStore();
        store.setStatus(WatchContracts.WatchConnectionStatus.STREAMING);
        store.acceptMeasurement(5.0, 70, new double[0], "aa", "watch");

        WatchContracts.WatchHeartRateSnapshot snapshot = store.snapshot(7.1);
        assertEquals(WatchContracts.WatchConnectionStatus.STALE, snapshot.status);
        assertTrue(snapshot.isStale(7.1, 2.0));
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectOutOfRangeBpm() {
        WatchSampleStore store = new WatchSampleStore();
        store.acceptMeasurement(1.0, 0, new double[0], "aa", "watch");
    }

    @Test
    public void malformedCounterIncrementsWithoutClearingHistory() {
        WatchSampleStore store = new WatchSampleStore();
        store.setStatus(WatchContracts.WatchConnectionStatus.STREAMING);
        store.acceptMeasurement(1.0, 80, new double[0], "aa", "watch");
        store.incrementMalformed();
        store.incrementMalformed();

        WatchContracts.WatchHeartRateSnapshot snapshot = store.snapshot(1.0);
        assertEquals(2, snapshot.malformedPackets);
        assertEquals(1, snapshot.samples.size());
        assertEquals(WatchContracts.WatchConnectionStatus.STREAMING, snapshot.status);
    }

    @Test
    public void setDevicesCopiesIntoSnapshot() {
        WatchSampleStore store = new WatchSampleStore();
        store.setDevices(Collections.singletonList(new WatchContracts.WatchDevice("1", "Huawei")));
        WatchContracts.WatchHeartRateSnapshot snapshot = store.snapshot(0.0);
        assertEquals(1, snapshot.devices.size());
        assertEquals("Huawei", snapshot.devices.get(0).name);
        assertNull(snapshot.latestSample);
    }
}
