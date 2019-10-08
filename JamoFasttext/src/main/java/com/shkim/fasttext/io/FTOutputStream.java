package com.shkim.fasttext.io;

import com.google.common.io.LittleEndianDataOutputStream;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.Charset;
import java.util.Objects;


/**
 * FastText output stream.
 * To write byte data in c++(linux) little endian order.
 * Covers only primitives.
 * @see com.google.common.io.LittleEndianDataOutputStream
 *
 * Created by @szuev on 26.10.2017.
 */
public class FTOutputStream extends FilterOutputStream {

    public FTOutputStream(OutputStream out) {
        super(wrap(out));
    }

    private static LittleEndianDataOutputStream wrap(OutputStream out) {
        Objects.requireNonNull(out, "Null output steam specified.");
        return out instanceof LittleEndianDataOutputStream ? (LittleEndianDataOutputStream) out : new LittleEndianDataOutputStream(out);
    }

    private LittleEndianDataOutputStream out() {
        return (LittleEndianDataOutputStream) out;
    }

    @Override
    public void write(byte[] b, int off, int len) throws IOException {
        out.write(b, off, len);
    }

    public void writeBoolean(boolean v) throws IOException {
        out().writeBoolean(v);
    }

    public void writeByte(int v) throws IOException {
        out().writeByte(v);
    }

    public void writeDouble(double v) throws IOException {
        out().writeDouble(v);
    }

    public void writeFloat(float v) throws IOException {
        out().writeFloat(v);
    }

    public void writeInt(int v) throws IOException {
        out().writeInt(v);
    }

    public void writeLong(long v) throws IOException {
        out().writeLong(v);
    }

    public void writeShort(int v) throws IOException {
        out().writeShort(v);
    }

    @Override
    public void close() throws IOException {
        out.close();
    }

    /**
     * Writes string to the output stream as byte array in specified encoding with '0' as end indicator.
     *
     * @param out     {@link OutputStream} any output stream, not null
     * @param str     String, not null
     * @param charset {@link Charset}, not null
     * @throws IOException if something goes wrong
     * @see FTInputStream#readString(InputStream, Charset)
     */
    public static void writeString(OutputStream out, String str, Charset charset) throws IOException {
        out.write(str.getBytes(charset));
        out.write(0);
    }
}
