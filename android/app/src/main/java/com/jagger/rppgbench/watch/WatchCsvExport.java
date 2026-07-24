package com.jagger.rppgbench.watch;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.util.List;

public final class WatchCsvExport {
    public static final String SAMPLES_FILE = "watch_heart_rate_samples.csv";
    public static final String ALIGNMENTS_FILE = "watch_rppg_alignments.csv";
    public static final String NOTES_FILE = "watch_export_notes.txt";

    private WatchCsvExport() {
    }

    public static void writeSessionArtifacts(
            File outputDirectory,
            List<WatchContracts.WatchHeartRateSample> samples,
            List<WatchContracts.WatchAlignmentResult> alignments,
            Double sessionStartMonotonicSec,
            String connectedDeviceLabel) throws IOException {
        if (outputDirectory == null) {
            throw new IllegalArgumentException("outputDirectory");
        }
        if (!outputDirectory.exists() && !outputDirectory.mkdirs()) {
            throw new IOException("could not create session directory: " + outputDirectory);
        }
        writeNotes(outputDirectory, connectedDeviceLabel);
        writeSamples(outputDirectory, samples, sessionStartMonotonicSec);
        writeAlignments(outputDirectory, alignments);
    }

    public static void writeNotes(File outputDirectory, String connectedDeviceLabel)
            throws IOException {
        String label =
                connectedDeviceLabel == null || connectedDeviceLabel.isEmpty()
                        ? "HUAWEI WATCH GT 5 Pro"
                        : connectedDeviceLabel;
        String body =
                "device_label="
                        + label
                        + "\n"
                        + "role=experimental_reference_not_medical_diagnosis\n"
                        + "timestamp_note=received_monotonic_sec is phone receive time, "
                        + "not watch internal sample time\n";
        writeText(new File(outputDirectory, NOTES_FILE), body);
    }

    public static void writeSamples(
            File outputDirectory,
            List<WatchContracts.WatchHeartRateSample> samples,
            Double sessionStartMonotonicSec) throws IOException {
        StringBuilder builder = new StringBuilder();
        builder.append(
                "device_id,device_name,received_monotonic_sec,session_timestamp_sec,bpm,"
                        + "rr_intervals_sec,parse_status\n");
        if (samples != null) {
            for (WatchContracts.WatchHeartRateSample sample : samples) {
                builder.append(csv(sample.deviceId)).append(',')
                        .append(csv(sample.deviceName)).append(',')
                        .append(number(sample.receivedMonotonicSec)).append(',');
                if (sessionStartMonotonicSec == null) {
                    builder.append(',');
                } else {
                    builder.append(number(sample.receivedMonotonicSec - sessionStartMonotonicSec))
                            .append(',');
                }
                builder.append(sample.bpm).append(',')
                        .append(csv(rrJson(sample.rrIntervalsSec))).append(',')
                        .append("VALID").append('\n');
            }
        }
        writeText(new File(outputDirectory, SAMPLES_FILE), builder.toString());
    }

    public static void writeAlignments(
            File outputDirectory, List<WatchContracts.WatchAlignmentResult> alignments)
            throws IOException {
        StringBuilder builder = new StringBuilder();
        builder.append(
                "start_sec,end_sec,rppg_bpm,watch_reference_bpm,signed_error_bpm,"
                        + "absolute_error_bpm,watch_sample_count,coverage_ratio,max_gap_sec,"
                        + "alignment_status\n");
        if (alignments != null) {
            for (WatchContracts.WatchAlignmentResult row : alignments) {
                builder.append(number(row.startSec)).append(',')
                        .append(number(row.endSec)).append(',')
                        .append(nullableNumber(row.rppgBpm)).append(',')
                        .append(nullableNumber(row.watchReferenceBpm)).append(',')
                        .append(nullableNumber(row.signedErrorBpm)).append(',')
                        .append(nullableNumber(row.absoluteErrorBpm)).append(',')
                        .append(row.watchSampleCount).append(',')
                        .append(number(row.coverageRatio)).append(',')
                        .append(nullableNumber(row.maxGapSec)).append(',')
                        .append(row.status.name()).append('\n');
            }
        }
        writeText(new File(outputDirectory, ALIGNMENTS_FILE), builder.toString());
    }

    private static String rrJson(double[] values) {
        StringBuilder builder = new StringBuilder("[");
        if (values != null) {
            for (int index = 0; index < values.length; index++) {
                if (index > 0) {
                    builder.append(',');
                }
                builder.append(number(values[index]));
            }
        }
        builder.append(']');
        return builder.toString();
    }

    private static String number(double value) {
        return Double.toString(value);
    }

    private static String nullableNumber(Double value) {
        return value == null ? "" : number(value);
    }

    private static String csv(String value) {
        if (value == null) {
            return "";
        }
        if (value.indexOf(',') >= 0 || value.indexOf('"') >= 0 || value.indexOf('\n') >= 0) {
            return '"' + value.replace("\"", "\"\"") + '"';
        }
        return value;
    }

    private static void writeText(File file, String body) throws IOException {
        try (Writer writer =
                new OutputStreamWriter(new FileOutputStream(file), StandardCharsets.UTF_8)) {
            writer.write(body);
        }
    }
}
