package com.jagger.rppgbench;

import java.io.File;

/** Immutable UI-to-runtime mapping for one deep-model choice. */
public final class DeepModelSelection {
    private static final DeepModelSelection[] OPTIONS = {
        new DeepModelSelection("关闭", "disabled", "深度模型", "", null),
        new DeepModelSelection(
                "TSCAN (UBFC)",
                "tscan",
                "TSCAN",
                "UBFC",
                "ubfc_tscan_full_lr3e-5_Epoch10.onnx"),
        new DeepModelSelection(
                "EfficientPhys (PURE)",
                "efficientphys",
                "EfficientPhys",
                "PURE",
                "efficientphys_pure.onnx")
    };

    public final String spinnerLabel;
    public final String canonicalName;
    public final String modelLabel;
    public final String datasetLabel;
    public final String filename;

    private DeepModelSelection(
            String spinnerLabel,
            String canonicalName,
            String modelLabel,
            String datasetLabel,
            String filename) {
        this.spinnerLabel = spinnerLabel;
        this.canonicalName = canonicalName;
        this.modelLabel = modelLabel;
        this.datasetLabel = datasetLabel;
        this.filename = filename;
    }

    public static DeepModelSelection[] spinnerOptions() {
        return OPTIONS.clone();
    }

    public static String[] spinnerLabels() {
        String[] labels = new String[OPTIONS.length];
        for (int index = 0; index < OPTIONS.length; ++index) {
            labels[index] = OPTIONS[index].spinnerLabel;
        }
        return labels;
    }

    public static DeepModelSelection fromSpinnerPosition(int position) {
        if (position < 0 || position >= OPTIONS.length) {
            throw new IllegalArgumentException("invalid deep model spinner position: " + position);
        }
        return OPTIONS[position];
    }

    public static DeepModelSelection fromCanonicalName(String canonicalName) {
        for (DeepModelSelection option : OPTIONS) {
            if (option.canonicalName.equals(canonicalName)) {
                return option;
            }
        }
        throw new IllegalArgumentException("unknown deep model: " + canonicalName);
    }

    public boolean enabled() {
        return filename != null;
    }

    public File modelFile(File modelDirectory) {
        if (!enabled()) {
            return null;
        }
        if (modelDirectory == null) {
            throw new IllegalArgumentException("model directory is required");
        }
        return new File(modelDirectory, filename);
    }

    public String displayLabel() {
        return datasetLabel.isEmpty() ? modelLabel : modelLabel + " (" + datasetLabel + ")";
    }
}
