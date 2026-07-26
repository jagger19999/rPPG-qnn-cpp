package com.jagger.rppgbench;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;

import java.io.File;
import org.junit.Test;

public final class DeepModelSelectionTest {
    @Test
    public void spinnerOptionsMapToCanonicalRuntimeContracts() {
        DeepModelSelection[] options = DeepModelSelection.spinnerOptions();

        assertEquals(3, options.length);
        assertSelection(options[0], "关闭", "disabled", "深度模型", "", null);
        assertSelection(
                options[1],
                "TSCAN (UBFC)",
                "tscan",
                "TSCAN",
                "UBFC",
                "ubfc_tscan_full_lr3e-5_Epoch10.onnx");
        assertSelection(
                options[2],
                "EfficientPhys (PURE)",
                "efficientphys",
                "EfficientPhys",
                "PURE",
                "efficientphys_pure.onnx");
    }

    @Test
    public void selectedModelResolvesOnlyItsAppPrivateFile() {
        File modelDirectory = new File("/app/files/models");
        assertNull(DeepModelSelection.fromSpinnerPosition(0).modelFile(modelDirectory));
        assertEquals(
                new File(modelDirectory, "ubfc_tscan_full_lr3e-5_Epoch10.onnx"),
                DeepModelSelection.fromSpinnerPosition(1).modelFile(modelDirectory));
        assertEquals(
                new File(modelDirectory, "efficientphys_pure.onnx"),
                DeepModelSelection.fromCanonicalName("efficientphys").modelFile(modelDirectory));
    }

    @Test
    public void invalidSpinnerAndCanonicalValuesAreRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> DeepModelSelection.fromSpinnerPosition(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> DeepModelSelection.fromSpinnerPosition(3));
        assertThrows(
                IllegalArgumentException.class,
                () -> DeepModelSelection.fromCanonicalName("TSCAN"));
    }

    private static void assertSelection(
            DeepModelSelection selection,
            String spinnerLabel,
            String canonicalName,
            String modelLabel,
            String datasetLabel,
            String filename) {
        assertEquals(spinnerLabel, selection.spinnerLabel);
        assertEquals(canonicalName, selection.canonicalName);
        assertEquals(modelLabel, selection.modelLabel);
        assertEquals(datasetLabel, selection.datasetLabel);
        assertEquals(filename, selection.filename);
    }
}
