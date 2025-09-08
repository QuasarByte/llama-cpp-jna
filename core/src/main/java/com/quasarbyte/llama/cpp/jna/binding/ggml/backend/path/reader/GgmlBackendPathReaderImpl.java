package com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.reader;

import com.quasarbyte.llama.cpp.jna.exception.LlamaCppJnaException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.quasarbyte.llama.cpp.jna.binding.ggml.backend.GgmlBackendConstants.GGML_BACKEND_PATH_ENV_VAR_NAME;
import static com.quasarbyte.llama.cpp.jna.binding.ggml.backend.GgmlBackendConstants.GGML_BACKEND_PATH_SYS_PROPERTY_NAME;

public class GgmlBackendPathReaderImpl implements GgmlBackendPathReader {

    private static final Logger logger = LoggerFactory.getLogger(GgmlBackendPathReaderImpl.class);

    @Override
    public String readBackendPath() {
        final String backendPath;

        String backendPathSystemProperty = System.getProperty(GGML_BACKEND_PATH_SYS_PROPERTY_NAME);
        String backendPathEnvironmentVariable = System.getenv(GGML_BACKEND_PATH_ENV_VAR_NAME);

        if (backendPathSystemProperty != null && !backendPathEnvironmentVariable.trim().isEmpty()) {
            backendPath = backendPathSystemProperty;
            logger.info("Set Ggml Backend System Property to backend path" + backendPathSystemProperty);
        } else if (backendPathEnvironmentVariable != null && !backendPathEnvironmentVariable.trim().isEmpty()) {
            backendPath = backendPathEnvironmentVariable;
            logger.info("Set Ggml Backend Environment Variable  to backend path" + backendPathEnvironmentVariable);
        } else {
            throw new LlamaCppJnaException(String.format("Unable to load Ggml Backend System Property. Java System Property '%s' or Environment Variable '%s'",
                    GGML_BACKEND_PATH_SYS_PROPERTY_NAME,
                    GGML_BACKEND_PATH_ENV_VAR_NAME));
        }
        return backendPath;
    }
}
