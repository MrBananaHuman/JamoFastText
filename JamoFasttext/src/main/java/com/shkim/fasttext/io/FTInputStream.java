package com.shkim.fasttext.io;

import com.google.common.io.LittleEndianDataInputStream;

import java.io.*;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * FastText InputStream.
 * To read byte data in cpp little endian style.
 * Covers only primitives.
 * @see com.google.common.io.LittleEndianDataInputStream
 *
 * Created by @szuev on 26.10.2017.
 */
public class FTInputStream extends FilterInputStream {

    public FTInputStream(InputStream in) {
        super(wrap(in));
    }

    private static LittleEndianDataInputStream wrap(InputStream in) {
        Objects.requireNonNull(in, "Null input stream specified");
        return in instanceof LittleEndianDataInputStream ? (LittleEndianDataInputStream) in : new LittleEndianDataInputStream(in);
    }

    private LittleEndianDataInputStream in() {
        return (LittleEndianDataInputStream) in;
    }

    public void readFully(byte[] b) throws IOException {
        in().readFully(b);
    }

    public void readFully(byte[] b, int off, int len) throws IOException {
        in().readFully(b, off, len);
    }

    public boolean readBoolean() throws IOException {
        return in().readBoolean();
    }

    public byte readByte() throws IOException {
        return in().readByte();
    }

    public int readUnsignedByte() throws IOException {
        return in().readUnsignedByte();
    }

    public short readShort() throws IOException {
        return in().readShort();
    }

    public int readUnsignedShort() throws IOException {
        return in().readUnsignedShort();
    }

    public char readChar() throws IOException {
        return in().readChar();
    }

    public int readInt() throws IOException {
        return in().readInt();
    }

    public long readLong() throws IOException {
        return in().readLong();
    }

    public float readFloat() throws IOException {
        return in().readFloat();
    }

    public double readDouble() throws IOException {
        return in().readDouble();
    }

    /**
     * Reads an array of bytes from input stream till specified end character
     *
     * @param in  {@link InputStream}
     * @param end byte
     * @return byte[]
     * @throws IOException and I/O error
     */
    public static byte[] readUpToByte(InputStream in, byte end) throws IOException {
        List<Integer> buff = new ArrayList<>(128);
        while (true) {
            int c = in.read();
            if (c == end) {
                break;
            }
            if (c == -1) throw new EOFException();
            buff.add(c);
        }
        byte[] res = new byte[buff.size()];
        for (int i = 0; i < buff.size(); i++) {
            res[i] = buff.get(i).byteValue();
        }
        return res;
    }

    /**
     * Reads string from output stream in specified encoding, byte '0' indicates the end of String
     *
     * @param in      {@link InputStream} any input stream, not null
     * @param charset {@link Charset}, not null
     * @return decoded String
     * @throws IOException if something goes wrong
     * @see FTOutputStream#writeString(OutputStream, String, Charset)
     */
    public static String readString(InputStream in, Charset charset) throws IOException {
        return new String(readUpToByte(in, (byte) 0), charset);
    }
}
