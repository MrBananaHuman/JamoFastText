package com.shkim.fasttext.io;

import java.io.PrintStream;

/**
 * Log printer, used during {@link ExtraFunction.fasttext.FastText} operations.
 * It is analogue of any other common java logger (e.g. log4j), but with only few log levels.
 * Note: methods {@link #trace}, {@link #debug}, {@link #info} do not perform switching to new line.
 * <p>
 * Created by @szuev on 08.12.2017.
 */
public interface PrintLogs {

    boolean isTraceEnabled();

    boolean isDebugEnabled();

    boolean isInfoEnabled();

    void trace(String msg, Object... args);

    void debug(String msg, Object... args);

    void info(String msg, Object... args);

    default boolean isEnabled() {
        return isInfoEnabled() || isDebugEnabled() || isTraceEnabled();
    }

    default void traceln(String msg, Object... args) {
        if (msg == null) return;
        trace(msg + System.lineSeparator(), args);
    }

    default void debugln(String msg, Object... args) {
        if (msg == null) return;
        debug(msg + System.lineSeparator(), args);
    }

    default void infoln(String msg, Object... args) {
        if (msg == null) return;
        info(msg + System.lineSeparator(), args);
    }

    @FunctionalInterface
    interface MessagePrinter {

        /**
         * Prints formatted message
         *
         * @param pattern the message pattern
         * @param args    array of arguments to paste into the message pattern
         * @throws NullPointerException             if string pattern is null
         * @throws java.util.IllegalFormatException if pattern and args do not match
         * @see String#format(String, Object...)
         * @see PrintStream#printf(String, Object...)
         */
        void printf(String pattern, Object... args);
    }

    /**
     * A leveled wrapper around {@link MessagePrinter}, the default {@link PrintLogs} implementation.
     */
    class Impl implements PrintLogs, MessagePrinter {
        private final Level level;
        private final MessagePrinter delegate;

        public Impl(Level level, MessagePrinter printer) {
            this.level = level;
            this.delegate = printer;
        }

        @Override
        public boolean isTraceEnabled() {
            return isGreaterOrEqual(Level.TRACE);
        }

        @Override
        public boolean isDebugEnabled() {
            return isGreaterOrEqual(Level.DEBUG);
        }

        @Override
        public boolean isInfoEnabled() {
            return isGreaterOrEqual(Level.INFO);
        }

        private boolean isGreaterOrEqual(Level other) {
            return this.level.compareTo(other) >= 0;
        }

        @Override
        public void printf(String pattern, Object... args) {
            if (pattern == null) return;
            delegate.printf(pattern, args);
        }

        @Override
        public boolean isEnabled() {
            return Level.NONE != level;
        }

        @Override
        public void trace(String msg, Object... args) {
            if (!isTraceEnabled()) return;
            printf(msg, args);
        }

        @Override
        public void debug(String msg, Object... args) {
            if (!isDebugEnabled()) return;
            printf(msg, args);
        }

        @Override
        public void info(String msg, Object... args) {
            if (!isInfoEnabled()) return;
            printf(msg, args);
        }

    }

    /**
     * The log-level enum and the factory to create {@link PrintLogs}
     */
    enum Level {
        NONE, INFO, DEBUG, TRACE, ALL,;

        public static Level at(int index) {
            try {
                return values()[index];
            } catch (IndexOutOfBoundsException e) {
                return index > 0 ? ALL : NONE;
            }
        }

        public PrintLogs createLogger(MessagePrinter printer) {
            return new Impl(this, printer);
        }

        public PrintLogs createLogger(PrintStream out) {
            return createLogger((s, a) -> {
                if (a.length == 0) {
                    out.print(s);
                } else {
                    out.printf(s, a);
                }
            });
        }
    }
}
