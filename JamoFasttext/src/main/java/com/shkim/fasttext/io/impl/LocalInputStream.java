package com.shkim.fasttext.io.impl;

import org.apache.commons.math3.util.FastMath;

import com.shkim.fasttext.io.ScrollableInputStream;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;
import java.util.Objects;

/**
 * The simple version of {@link org.apache.hadoop.fs.RawLocalFileSystem.LocalFSFileInputStream}
 * Created by @szuev on 30.10.2017.
 */
@SuppressWarnings("WeakerAccess")
public class LocalInputStream extends ScrollableInputStream {

    private final RandomAccessFile rfa;
    private long size;

    public LocalInputStream(Path file) throws IOException {
        this.rfa = new RandomAccessFile(file.toFile(), "r");
        this.size = rfa.length(); // return Files.size(path);
        //this.fis = new FileInputStream(file.toFile());
    }

    @Override
    public void seek(long pos) throws IOException {
        if (pos < 0) {
            throw new IllegalArgumentException("Negative position: " + pos);
        }
        rfa.seek(pos);
    }

    @Override
    public long getPos() throws IOException {
        return rfa.getFilePointer();
    }

    @Override
    public long getLen() {
        return size;
    }

    @Override
    public int available() throws IOException {
        long res = FastMath.max(0L, getLen() - getPos());
        return res > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) res;
    }

    @Override
    public void close() throws IOException {
        rfa.close();
    }

    @Override
    public boolean markSupported() {
        return false;
    }

    @Override
    public int read() throws IOException {
        return rfa.read();
    }

    @SuppressWarnings("NullableProblems")
    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        return rfa.read(Objects.requireNonNull(b, "Null buff"), off, len);
    }

    @Override
    public long skip(long n) throws IOException {
        if (n >= Integer.MAX_VALUE) throw new IllegalArgumentException("Out of range");
        return rfa.skipBytes((int) n);
    }

}
