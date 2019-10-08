package com.shkim.fasttext.io;

import java.io.IOException;
import java.io.InputStream;

/**
 * see {@link org.apache.hadoop.fs.Seekable}
 * NOTE: the file should not be changed during operation with this stream
 * Created by @szuev on 30.10.2017.
 */
public abstract class ScrollableInputStream extends InputStream {

    /**
     * Seeks to the given offset in bytes from the start of the stream.
     * The next read() will be from that location.
     * Can't seek past the end of the file.
     *
     * @param bytes
     * @throws IOException
     */
    public abstract void seek(long bytes) throws IOException;

    /**
     * Returns the current offset from the start of the file
     *
     * @return
     * @throws IOException
     */
    public abstract long getPos() throws IOException;

    /**
     * @return
     * @throws IOException
     */
    public abstract long getLen() throws IOException;

    /**
     * @return
     * @throws IOException
     */
    public boolean isEnd() throws IOException {
        return getLen() == getPos();
    }

    /**
     * @return
     * @throws IOException
     */
    public boolean isStart() throws IOException {
        return getPos() == 0;
    }
}
