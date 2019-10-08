package com.shkim.fasttext.io;

import java.io.Closeable;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * The buffered stream reader which allows to read word-tokens from any binary {@link InputStream input stream}.
 * Not thread-safe.
 * Expected to be faster then standard {@link java.io.BufferedReader BufferedReader}.
 * <p>
 * Created by @szuev on 20.12.2017.
 */
public class WordReader implements Closeable {
    public static final int END = -129;

    private final Charset charset;
    protected final InputStream in;
    private final String newLine;
    private final byte[] delimiters;
    private final byte[] buffer;

    private int index;
    private int res;
    private int start;
    private byte[] tmp;

    /**
     * The main constructor.
     *
     * @param in            {@link InputStream} the input stream to wrap
     * @param charset       {@link Charset} encoding
     * @param bufferSize    the size of buffer
     * @param newLineSymbol String to return on new line
     * @param delimiters    sequence of delimiters as byte array, not empty, the first element will be treated as line separator symbol (e.g. '\n')
     */
    public WordReader(InputStream in, Charset charset, int bufferSize, String newLineSymbol, byte... delimiters) {
        this.charset = Objects.requireNonNull(charset, "Null charset");
        this.in = Objects.requireNonNull(in, "Null input stream");
        if (bufferSize <= 0) {
            throw new IllegalArgumentException("Buffer size must be positive number");
        }
        this.buffer = new byte[bufferSize];
        this.newLine = Objects.requireNonNull(newLineSymbol, "New line symbol can not be empty");
        if (delimiters.length == 0) {
            throw new IllegalArgumentException("No delimiters specified.");
        }
        this.delimiters = delimiters;
    }

    public WordReader(InputStream in, Charset charset, int bufferSize, String newLineSymbol, String delimiters) {
        this(in, charset, bufferSize, newLineSymbol, Objects.requireNonNull(delimiters, "Null delimiters").getBytes(charset));
    }

    public WordReader(InputStream in, String newLineSymbol, String delimiters) {
        this(in, StandardCharsets.UTF_8, 8 * 1024, newLineSymbol, delimiters);
    }

    /**
     * Constructs a char token reader with '\n' as line separator and space ('\u0020') as token separator.
     *
     * @param in {@link InputStream} to wrap
     */
    public WordReader(InputStream in) {
        this(in, "\n", "\n ");
    }

    /**
     * Reads the next byte of data from the underling input stream.
     *
     * @return byte, int from -128 to 127 or {@link #END -129} in case of stream end.
     * @throws IOException if some I/O error occurs
     * @see InputStream#read(byte[], int, int)
     */
    public int nextByte() throws IOException {
        if (index == buffer.length || res == 0) {
            if (start != 0) {
                if (buffer.length <= start) {
                    start = 0;
                } else {
                    tmp = new byte[buffer.length - start];
                    System.arraycopy(buffer, start, tmp, 0, tmp.length);
                }
            }
            res = in.read(buffer, 0, buffer.length);
            if (res == -1) {
                return END;
            }
            index = 0;
        }
        if (index < res) {
            return buffer[index++];
        }
        return END;
    }

    /**
     * Reads next word token from the underling input stream.
     *
     * @return String or null in case of end of stream
     * @throws IOException if some I/O error occurs
     */
    public String nextWord() throws IOException {
        this.start = index;
        int len = 0;
        int b;
        while ((b = nextByte()) != END) {
            if (!isDelimiter(b)) {
                len++;
                continue;
            }
            if (len == 0) {
                if (isNewLine(b)) {
                    return newLine;
                }
                start++;
            } else {
                if (isNewLine(b)) {
                    --index;
                }
                return makeString(len);
            }
        }
        return len == 0 ? null : makeString(len);
    }

    /**
     * Resets the state variables.
     */
    protected void reset() {
        start = index = res = 0;
        tmp = null;
    }

    /**
     * Answers if the end of stream is reached.
     *
     * @return boolean
     */
    protected boolean isEnd() {
        return res == -1 || res != 0 && res < buffer.length && index >= res;
    }

    protected boolean isDelimiter(int b) {
        for (byte i : delimiters) {
            if (b == i) return true;
        }
        return false;
    }

    protected boolean isNewLine(int b) {
        return delimiters[0] == b;
    }

    private String makeString(int len) {
        String str = "</s>";
        if(len <= buffer.length) {
        	if (start > index) {
                byte[] bytes = new byte[len];
                if (len <= tmp.length) {          	
                    System.arraycopy(tmp, 0, bytes, 0, len);
                    this.start = start + len;
                } else {
                    System.arraycopy(tmp, 0, bytes, 0, tmp.length);              
                    System.arraycopy(buffer, 0, bytes, tmp.length, len - tmp.length);
                }
                str = new String(bytes, charset);
            } else {
            	str = new String(buffer, start, len, charset);
            }
        }
        
        return str;
    }

    @Override
    public void close() throws IOException {
        in.close();
    }

    public Stream<String> words() {
        Iterator<String> iter = new Iterator<String>() {
            String next = null;

            @Override
            public boolean hasNext() {
                if (next != null) {
                    return true;
                } else {
                    try {
                        next = nextWord();
                        return next != null;
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }
            }

            @Override
            public String next() {
                if (next != null || hasNext()) {
                    String line = this.next;
                    this.next = null;
                    return line;
                } else {
                    throw new NoSuchElementException();
                }
            }
        };
        return StreamSupport.stream(Spliterators.spliteratorUnknownSize(iter, Spliterator.ORDERED | Spliterator.NONNULL), false);
    }
}
