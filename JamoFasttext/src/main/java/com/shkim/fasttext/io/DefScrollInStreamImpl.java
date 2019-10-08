package com.shkim.fasttext.io;

import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/**
 * Internal, the part of {@link IOStreams}.
 * Default implementation of {@link ScrollableInputStream}.
 * <p>
 * Created by @szuev on 27.12.2017.
 */
class DefScrollInStreamImpl extends ScrollableInputStream {
    private static final int BUFF_SIZE = 8 * 1024;

    private final IOStreams fs;
    private final String uri;

    private long size = -1;
    private long position;
    private InputStream in;

    DefScrollInStreamImpl(String uri, IOStreams fs) {
        this.fs = Objects.requireNonNull(fs, "Null fs");
        this.uri = Objects.requireNonNull(uri, "Null uri");
    }

    private InputStream open() throws IOException {
        return fs.openInput(uri);
    }

    private InputStream fetch() throws IOException {
        return in == null ? in = open() : in;
    }

    private synchronized void clear() {
        in = null;
        position = 0;
    }

    @Override
    public synchronized void seek(long bytes) throws IOException {
        if (in != null && bytes >= position) {
            long x = bytes - position;
            position += in.skip(x);
            return;
        }
        clear();
        position = fetch().skip(bytes);
    }

    @Override
    public long getPos() {
        return position;
    }

    @Override
    public long getLen() throws IOException {
        if (size != -1) return size;
        synchronized (this) {
            if (size != -1) return size;
            byte[] tmp = new byte[BUFF_SIZE];
            try (InputStream in = open()) {
                long size = 0;
                int res;
                while ((res = in.read(tmp)) != -1) {
                    size += res;
                }
                return this.size = size;
            }
        }
    }

    @Override
    public synchronized int read() throws IOException {
        int res = fetch().read();
        if (res == -1) {
            return -1;
        }
        position++;
        return res;
    }

    @Override
    public void close() throws IOException {
        if (in != null) in.close();
        clear();
    }

}
