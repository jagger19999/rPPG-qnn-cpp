package com.jagger.rppgbench.watch;

public final class HeartRateParser {
    public static final class Parsed {
        public final int bpm;
        public final double[] rrIntervalsSec;

        public Parsed(int bpm, double[] rrIntervalsSec) {
            this.bpm = bpm;
            this.rrIntervalsSec = rrIntervalsSec;
        }
    }

    private HeartRateParser() {
    }

    public static Parsed parse(byte[] data) {
        if (data == null || data.length < 2) {
            throw new IllegalArgumentException("heart rate packet is truncated");
        }
        int flags = data[0] & 0xFF;
        int cursor = 1;
        int bpm;
        if ((flags & 0x01) != 0) {
            if (data.length < 3) {
                throw new IllegalArgumentException("16-bit heart rate packet is truncated");
            }
            bpm = (data[cursor] & 0xFF) | ((data[cursor + 1] & 0xFF) << 8);
            cursor += 2;
        } else {
            bpm = data[cursor] & 0xFF;
            cursor += 1;
        }
        if ((flags & 0x08) != 0) {
            cursor += 2;
            if (cursor > data.length) {
                throw new IllegalArgumentException("energy expended field is truncated");
            }
        }
        double[] rr = new double[0];
        if ((flags & 0x10) != 0) {
            if (((data.length - cursor) & 1) != 0) {
                throw new IllegalArgumentException("RR interval field is truncated");
            }
            int count = (data.length - cursor) / 2;
            rr = new double[count];
            for (int i = 0; i < count; i++) {
                int raw = (data[cursor] & 0xFF) | ((data[cursor + 1] & 0xFF) << 8);
                cursor += 2;
                rr[i] = raw / 1024.0;
            }
        }
        return new Parsed(bpm, rr);
    }
}
