package com.jagger.rppgbench;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public final class ModelIntegrity {
    public static final class Spec {
        public final String filename;
        public final String sha256;

        private Spec(String filename, String sha256) {
            this.filename = filename;
            this.sha256 = sha256;
        }
    }

    private static final Spec TSCAN =
            new Spec(
                    "ubfc_tscan_full_lr3e-5_Epoch10.onnx",
                    "342a3c8033dda9ab154e85d5a4e2a876a6461648b7fcb27c46a7023e662bcc64");
    private static final Spec EFFICIENTPHYS =
            new Spec(
                    "efficientphys_pure.onnx",
                    "c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d");

    private ModelIntegrity() {}

    public static Spec specFor(String deepModel) {
        switch (deepModel) {
            case "tscan":
                return TSCAN;
            case "efficientphys":
                return EFFICIENTPHYS;
            case "disabled":
                return null;
            default:
                throw new IllegalArgumentException("unknown deep model: " + deepModel);
        }
    }

    public static Spec specFor(DeepModelSelection selection) {
        if (selection == null) {
            throw new IllegalArgumentException("deep model selection is required");
        }
        Spec spec = specFor(selection.canonicalName);
        if (spec != null && !spec.filename.equals(selection.filename)) {
            throw new IllegalStateException("deep model selection filename is inconsistent");
        }
        return spec;
    }

    public static File modelFile(File modelDirectory, String deepModel) {
        Spec spec = specFor(deepModel);
        return spec == null ? null : new File(modelDirectory, spec.filename);
    }

    public static File modelFile(File modelDirectory, DeepModelSelection selection) {
        specFor(selection);
        return selection.modelFile(modelDirectory);
    }

    static void requireExpectedFilename(Spec spec, File model) {
        if (model == null || !spec.filename.equals(model.getName())) {
            throw new IllegalArgumentException("selected model filename must be " + spec.filename);
        }
    }

    static void verifySha256(String expected, InputStream input) throws IOException {
        String actual = sha256(input);
        if (!expected.equals(actual)) {
            throw new IllegalArgumentException("selected model SHA-256 mismatch: " + actual);
        }
    }

    static String sha256(InputStream input) throws IOException {
        final MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
        byte[] buffer = new byte[1024 * 1024];
        int count;
        while ((count = input.read(buffer)) != -1) {
            digest.update(buffer, 0, count);
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) {
            hex.append(Character.forDigit((value >>> 4) & 0x0f, 16));
            hex.append(Character.forDigit(value & 0x0f, 16));
        }
        return hex.toString();
    }
}
