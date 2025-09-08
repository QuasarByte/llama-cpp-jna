package com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader;

import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibrary;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.reader.GgmlBackendPathReader;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.validator.GgmlBackendPathValidator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class GgmlBackendLoaderImpl implements GgmlBackendLoader {

    private static final Logger logger = LoggerFactory.getLogger(GgmlBackendLoaderImpl.class);

    private final GgmlLibrary ggmlLibrary;
    private final GgmlBackendPathReader ggmlBackendPathReader;
    private final GgmlBackendPathValidator ggmlBackendPathValidator;

    public GgmlBackendLoaderImpl(GgmlLibrary ggmlLibrary,
                                 GgmlBackendPathReader ggmlBackendPathReader,
                                 GgmlBackendPathValidator ggmlBackendPathValidator) {
        this.ggmlLibrary = ggmlLibrary;
        this.ggmlBackendPathReader = ggmlBackendPathReader;
        this.ggmlBackendPathValidator = ggmlBackendPathValidator;
    }

    @Override
    public void loadBackend() {
        String backendPath = ggmlBackendPathReader.readBackendPath();
        ggmlBackendPathValidator.validateBackendPath(backendPath);
        ggmlLibrary.ggml_backend_load_all_from_path(backendPath);
    }

}
