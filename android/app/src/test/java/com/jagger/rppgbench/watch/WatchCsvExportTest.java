package com.jagger.rppgbench.watch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class WatchCsvExportTest {
    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void writesSampleAndAlignmentHeadersAndRows() throws Exception {
        File directory = temporaryFolder.newFolder("session");
        WatchContracts.WatchHeartRateSample sample =
                new WatchContracts.WatchHeartRateSample(
                        100.5, 72, new double[] {0.8}, "aa:bb", "HUAWEI WATCH GT 5 Pro");
        WatchContracts.WatchAlignmentResult alignment =
                new WatchContracts.WatchAlignmentResult(
                        0.0,
                        10.0,
                        74.0,
                        72.0,
                        2.0,
                        2.0,
                        4,
                        0.85,
                        1.5,
                        WatchContracts.WatchAlignmentStatus.ALIGNED);

        WatchCsvExport.writeSessionArtifacts(
                directory,
                Collections.singletonList(sample),
                Collections.singletonList(alignment),
                100.0,
                "HUAWEI WATCH GT 5 Pro");

        String samples =
                new String(
                        Files.readAllBytes(new File(directory, WatchCsvExport.SAMPLES_FILE).toPath()),
                        StandardCharsets.UTF_8);
        String alignments =
                new String(
                        Files.readAllBytes(
                                new File(directory, WatchCsvExport.ALIGNMENTS_FILE).toPath()),
                        StandardCharsets.UTF_8);
        String notes =
                new String(
                        Files.readAllBytes(new File(directory, WatchCsvExport.NOTES_FILE).toPath()),
                        StandardCharsets.UTF_8);

        assertTrue(samples.startsWith(
                "device_id,device_name,received_monotonic_sec,session_timestamp_sec,bpm,"
                        + "rr_intervals_sec,parse_status\n"));
        assertTrue(samples.contains("aa:bb,HUAWEI WATCH GT 5 Pro,100.5,0.5,72,[0.8],VALID"));
        assertTrue(alignments.startsWith(
                "start_sec,end_sec,rppg_bpm,watch_reference_bpm,signed_error_bpm,"
                        + "absolute_error_bpm,watch_sample_count,coverage_ratio,max_gap_sec,"
                        + "alignment_status\n"));
        assertTrue(alignments.contains("0.0,10.0,74.0,72.0,2.0,2.0,4,0.85,1.5,ALIGNED"));
        assertTrue(notes.contains("experimental_reference_not_medical_diagnosis"));
        assertTrue(notes.contains("phone receive time"));
        assertEquals(2, samples.split("\n").length);
        assertEquals(2, alignments.split("\n").length);
    }

    @Test
    public void emptyListsStillWriteHeaders() throws Exception {
        File directory = temporaryFolder.newFolder("empty");
        WatchCsvExport.writeSessionArtifacts(
                directory, Arrays.asList(), Arrays.asList(), null, null);
        String samples =
                new String(
                        Files.readAllBytes(new File(directory, WatchCsvExport.SAMPLES_FILE).toPath()),
                        StandardCharsets.UTF_8);
        assertEquals(1, samples.trim().split("\n").length);
    }
}
