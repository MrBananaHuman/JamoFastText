package com.shkim.fasttext.module;

import org.apache.commons.lang.StringUtils;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Arrays;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.stream.Collectors;

/**
 * Class to measure time of events in runtime to gather statistics.
 * TODO: it's temporary and will be removed.
 * <p>
 * Created by @szuev on 25.12.2017.
 */
public enum Events {
    GET_FILE_SIZE,
    READ_DICT,
    IN_MATRIX_CREATE,
    OUT_MATRIX_CREATE,

    FILE_SEEK,
    DIC_GET_LINE,
    TRAIN_CALC,
    MODEL_UPDATE,
    MODEL_COMPUTE_HIDDEN,
    MODEL_LOSS_CALC,
    MODEL_GRAD_MUL,
    MODEL_INPUT_ADD_ROW,
    CREATE_RES_MODEL,
    TRAIN,
    SAVE_BIN,
    ALL;

    // disabled by default
    private static final boolean DISABLED = !Boolean.parseBoolean(System.getProperty("events", "false"));

    private ThreadLocal<Instant> start = new ThreadLocal<>();
    private ConcurrentLinkedQueue<Long> times = new ConcurrentLinkedQueue<>();

    public static boolean isDisabled() {
        return DISABLED;
    }

    public void start() {
        if (DISABLED) return;
        Instant now = Instant.now();
        if (start.get() != null) throw new IllegalStateException();
        start.set(now);
    }

    public void end() {
        if (DISABLED) return;
        Instant now = Instant.now();
        Instant start = this.start.get();
        if (start == null) throw new IllegalStateException();
        this.start.set(null);
        long time = ChronoUnit.MICROS.between(start, now);
        times.add(time);
    }

    public int size() {
        return times.size();
    }

    public double average() {
        return times.stream().mapToDouble(t -> t).average().orElse(Double.NaN) / 1_000_000;
    }

    public double sum() {
        return times.stream().mapToLong(t -> t).sum() / 1_000_000d;
    }

    @Override
    public String toString() {
        return StringUtils.rightPad(name(), 40) +
                StringUtils.rightPad(String.valueOf(size()), 15) +
                StringUtils.rightPad(String.valueOf(average()), 25) +
                StringUtils.rightPad(String.valueOf(sum()), 20);
    }

    public static String print() {
        if (DISABLED) return null;
        return Arrays.stream(values()).map(Events::toString).collect(Collectors.joining("\n"));
    }
}
