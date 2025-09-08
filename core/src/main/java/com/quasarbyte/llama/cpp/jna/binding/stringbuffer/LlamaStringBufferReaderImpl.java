package com.quasarbyte.llama.cpp.jna.binding.stringbuffer;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

public class LlamaStringBufferReaderImpl implements LlamaStringBufferReader {

    private static final Logger logger = LoggerFactory.getLogger(LlamaStringBufferReaderImpl.class);

    @Override
    public String readString(byte[] buffer, int length) {
        return readString(buffer, length, StandardCharsets.UTF_8);
    }

    @Override
    public String readString(byte[] buffer, int length, Charset charset) {
        Objects.requireNonNull(buffer);
        Objects.requireNonNull(charset);

        if (length < 0) {
            logger.error("buffer length less than {}", 0);
            throw new LlamaCppJnaException(String.format("buffer length less than %d", length));
        }

        if (buffer.length < length) {
            logger.error("buffer length less than length, buffer length: {}, length {}", buffer.length, length);
            throw new LlamaCppJnaException(String.format("buffer length less than length, buffer length: %d, length %d", buffer.length, length));
        }

        if (length == 0) {
            return "";
        }

        return new String(buffer, 0, length, charset);
    }
}
