package com.jagger.rppgbench;

import android.os.ParcelFileDescriptor;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public final class PreparedModel implements AutoCloseable {
    private final ParcelFileDescriptor descriptor;
    private final String nativePath;

    private PreparedModel(ParcelFileDescriptor descriptor, String nativePath) {
        this.descriptor = descriptor;
        this.nativePath = nativePath;
    }

    public static PreparedModel prepare(DeepModelSelection selection, File model)
            throws IOException {
        ModelIntegrity.Spec spec = ModelIntegrity.specFor(selection);
        if (spec == null) {
            if (model != null) {
                throw new IllegalArgumentException("disabled deep model must not have a model file");
            }
            return new PreparedModel(null, "");
        }
        ModelIntegrity.requireExpectedFilename(spec, model);

        ParcelFileDescriptor stableDescriptor =
                ParcelFileDescriptor.open(model, ParcelFileDescriptor.MODE_READ_ONLY);
        boolean prepared = false;
        try {
            ParcelFileDescriptor hashDescriptor =
                    ParcelFileDescriptor.dup(stableDescriptor.getFileDescriptor());
            try (InputStream input = new ParcelFileDescriptor.AutoCloseInputStream(hashDescriptor)) {
                ModelIntegrity.verifySha256(spec.sha256, input);
            }
            PreparedModel result =
                    new PreparedModel(
                            stableDescriptor, "/proc/self/fd/" + stableDescriptor.getFd());
            prepared = true;
            return result;
        } finally {
            if (!prepared) {
                stableDescriptor.close();
            }
        }
    }

    public String nativePath() {
        return nativePath;
    }

    @Override
    public void close() throws IOException {
        if (descriptor != null) {
            descriptor.close();
        }
    }
}
