package com.jagger.rppgbench;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;

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

    public static File modelFile(File modelDirectory, String deepModel) {
        Spec spec = specFor(deepModel);
        return spec == null ? null : new File(modelDirectory, spec.filename);
    }

    public static void verify(String deepModel, File model) throws IOException {
        Spec spec = specFor(deepModel);
        if (spec == null) {
            if (model != null) {
                throw new IllegalArgumentException("disabled deep model must not have a model file");
            }
            return;
        }
        if (model == null || !spec.filename.equals(model.getName())) {
            throw new IllegalArgumentException(
                    "selected " + deepModel + " model filename must be " + spec.filename);
        }
        if (!model.isFile()) {
            throw new IllegalArgumentException("selected model file is missing: " + model);
        }
        String actual = sha256(model);
        if (!spec.sha256.equals(actual)) {
            throw new IllegalArgumentException(
                    "selected " + deepModel + " model SHA-256 mismatch: " + actual);
        }
    }

    private static String sha256(File file) throws IOException {
        final MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 is unavailable", error);
        }
        byte[] buffer = new byte[1024 * 1024];
        try (InputStream input = new FileInputStream(file)) {
            int count;
            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
        }
        StringBuilder hex = new StringBuilder(64);
        for (byte value : digest.digest()) {
            hex.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return hex.toString();
    }
}
