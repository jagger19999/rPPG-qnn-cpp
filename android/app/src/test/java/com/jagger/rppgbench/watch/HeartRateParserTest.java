package com.jagger.rppgbench.watch;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public final class HeartRateParserTest {
    @Test
    public void parse8BitBpm() {
        HeartRateParser.Parsed parsed = HeartRateParser.parse(new byte[] {0x00, 72});
        assertEquals(72, parsed.bpm);
        assertEquals(0, parsed.rrIntervalsSec.length);
    }

    @Test
    public void parse16BitBpmAndRr() {
        HeartRateParser.Parsed parsed =
                HeartRateParser.parse(new byte[] {0x11, 44, 1, 0, 4, 32, 4});
        assertEquals(300, parsed.bpm);
        assertEquals(2, parsed.rrIntervalsSec.length);
        assertEquals(1.0, parsed.rrIntervalsSec[0], 1e-9);
        assertEquals(1.03125, parsed.rrIntervalsSec[1], 1e-9);
    }

    @Test
    public void parseEnergyExpendedBeforeRr() {
        HeartRateParser.Parsed parsed =
                HeartRateParser.parse(new byte[] {0x18, 72, 10, 0, 0, 4});
        assertEquals(72, parsed.bpm);
        assertEquals(1.0, parsed.rrIntervalsSec[0], 1e-9);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectTruncated() {
        HeartRateParser.parse(new byte[] {0x01, 0x48});
    }
}
