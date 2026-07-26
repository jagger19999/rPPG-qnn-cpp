package com.jagger.rppgbench;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public final class ModelIntegrityTest {
    @Rule public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void pinsExactModelFilenamesAndHashes() {
        ModelIntegrity.Spec tscan = ModelIntegrity.specFor("tscan");
        assertEquals("ubfc_tscan_full_lr3e-5_Epoch10.onnx", tscan.filename);
        assertEquals(
                "342a3c8033dda9ab154e85d5a4e2a876a6461648b7fcb27c46a7023e662bcc64",
                tscan.sha256);

        ModelIntegrity.Spec efficientPhys = ModelIntegrity.specFor("efficientphys");
        assertEquals("efficientphys_pure.onnx", efficientPhys.filename);
        assertEquals(
                "c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d",
                efficientPhys.sha256);
    }

    @Test
    public void rejectsWrongFilenameBeforeHashing() throws Exception {
        File wrong = temporaryFolder.newFile("renamed.onnx");
        Files.write(wrong.toPath(), "not a model".getBytes(StandardCharsets.UTF_8));
        try {
            ModelIntegrity.verify("tscan", wrong);
            fail("expected filename rejection");
        } catch (IllegalArgumentException error) {
            assertTrue(error.getMessage().contains("filename"));
        }
    }

    @Test
    public void rejectsHashMismatch() throws Exception {
        File model = temporaryFolder.newFile("ubfc_tscan_full_lr3e-5_Epoch10.onnx");
        Files.write(model.toPath(), "not the pinned model".getBytes(StandardCharsets.UTF_8));
        try {
            ModelIntegrity.verify("tscan", model);
            fail("expected SHA-256 rejection");
        } catch (IllegalArgumentException error) {
            assertTrue(error.getMessage().contains("SHA-256"));
        }
    }

    @Test
    public void disabledNeedsNoModelFile() throws Exception {
        ModelIntegrity.verify("disabled", null);
    }
}
