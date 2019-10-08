package com.shkim.fasttext.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.InvalidPathException;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * Abstract factory to create {@link java.io.InputStream} and {@link java.io.OutputStream} with the same nature
 * depending on the encapsulated file system.
 * <p>
 * Created by @szuev on 24.10.2017.
 */
public interface IOStreams {

    /**
     * Opens or creates a file, returning an output stream that may be used to write bytes to the file.
     * The resulting stream will not be buffered. The stream will be safe for access by multiple concurrent threads.
     * Truncates file if it exists.
     * May create parents of the file if it is possible.
     *
     * @param uri, the reference to the file.
     * @return {@link OutputStream} dependent on encapsulated file system.
     * @throws IOException if it is not possible to create new file or read existing.
     */
    OutputStream createOutput(String uri) throws IOException;

    /**
     * Opens a file, returning an input stream to read from the file.
     * The stream will not be buffered, and is not required to support the {@link InputStream#mark mark} or {@link InputStream#reset reset} methods.
     * The stream will be safe for access by multiple concurrent threads.
     *
     * @param uri, the reference to the file
     * @return {@link InputStream} dependent on encapsulated file system.
     * @throws IOException if something wrong, e.g. no file found.
     */
    InputStream openInput(String uri) throws IOException;

    /**
     * Checks that the specified URI is good enough to be read.
     * May access the file system.
     *
     * @param uri, the file URI
     * @return true in case file can be read.
     */
    default boolean canRead(String uri) {
        return true;
    }

    /**
     * Checks the specified URI is good enough to write new file.
     * May access the file system.
     *
     * @param uri, the file URI
     * @return true if no errors expected when writing a file.
     */
    default boolean canWrite(String uri) {
        return true;
    }

    /**
     * Opens a file to read with seek supporting.
     *
     * @param uri, the file URI
     * @return {@link ScrollableInputStream}
     * @throws IOException if I/O error occurs
     */
    default ScrollableInputStream openScrollable(String uri) throws IOException {
        return new DefScrollInStreamImpl(uri, this);
    }

    /**
     * Retrieves the file size.
     * @param uri, the file URI
     * @return long, the size in bytes
     * @throws IOException if I/O error occurs
     */
    default long size(String uri) throws IOException {
        return openScrollable(uri).getLen();
    }

    /**
     * Makes an URI from String.
     *
     * @param uri String, file-ref
     * @return {@link URI}
     * @throws NullPointerException     if null uri
     * @throws IllegalArgumentException if wrong uri
     */
    static URI toURI(String uri) {
        try {
            return new URI(Objects.requireNonNull(uri, "Null uri"));
        } catch (URISyntaxException u) {
            try {
                return Paths.get(uri).toUri();
            } catch (InvalidPathException i) {
                u.addSuppressed(i);
            }
            throw new IllegalArgumentException("Wrong file-ref <" + uri + ">", u);
        }
    }
}
