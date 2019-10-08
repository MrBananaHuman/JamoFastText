package com.shkim.fasttext.io.impl;

import org.apache.commons.lang.StringUtils;

import com.shkim.fasttext.io.IOStreams;
import com.shkim.fasttext.io.ScrollableInputStream;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Created by @szuev on 30.10.2017.
 */
public class LocalIOStreams implements IOStreams {

    @Override
    public OutputStream createOutput(String uri) throws IOException {
        prepareForWrite(uri);
        return Files.newOutputStream(Paths.get(uri));
    }

    @Override
    public InputStream openInput(String uri) throws IOException {
        return Files.newInputStream(Paths.get(uri));
    }

    @Override
    public boolean canRead(String uri) {
        if (StringUtils.isEmpty(uri)) return false;
        Path file = Paths.get(uri);
        return Files.isRegularFile(file) && Files.isReadable(file);
    }

    @Override
    public boolean canWrite(String uri) { // '-' is reserved for std::out
        if (StringUtils.isEmpty(uri) || "-".equals(uri)) {
            return false;
        }
        Path parent = getParent(uri);
        return parent != null;
    }

    private Path getParent(String path) {
        Path file = Paths.get(path);
        Path res = file.getParent();
        if (res == null) {
            res = file.toAbsolutePath().getParent();
        }
        return res;
    }

    /**
     * Prepares the file to write: creates all parents and deletes previous version of file.
     *
     * @param uri String, path identifier to file entity.
     * @throws IOException if something goes wrong while preparation.
     */
    public void prepareForWrite(String uri) throws IOException {
        Path parent = getParent(uri);
        if (parent == null) throw new IOException("No parent for " + uri);
        Files.createDirectories(parent);
        Files.deleteIfExists(parent.resolve(uri));
    }

    @Override
    public ScrollableInputStream openScrollable(String uri) throws IOException {
        return new LocalInputStream(Paths.get(uri));
    }

    @Override
    public long size(String uri) throws IOException {
        return Files.size(Paths.get(uri));
    }

}
