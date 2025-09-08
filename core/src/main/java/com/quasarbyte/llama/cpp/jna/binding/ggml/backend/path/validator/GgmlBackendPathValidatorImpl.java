package com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.validator;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class GgmlBackendPathValidatorImpl implements GgmlBackendPathValidator {

    private static final Logger logger = LoggerFactory.getLogger(GgmlBackendPathValidatorImpl.class);

    @Override
    public void validateBackendPath(String path) {

        logger.debug("Validating backend path {}", path);

        if (path == null) {
            throw new LlamaCppJnaException("GGML backend path can not be null");
        } else if (path.trim().isEmpty()) {
            throw new LlamaCppJnaException("GGML backend path can not be blank");
        }

        Path p = Paths.get(path);

        if (Files.exists(p)) {

            if (!Files.isDirectory(p)) {
                throw new LlamaCppJnaException("The GGML backend path should be a directory, not a file");
            }

        } else {
            throw new LlamaCppJnaException("GGML backend path does not exist");
        }

    }

}
