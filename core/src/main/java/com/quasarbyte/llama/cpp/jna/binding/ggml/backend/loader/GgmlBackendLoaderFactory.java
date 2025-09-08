package com.quasarbyte.llama.cpp.jna.binding.ggml.backend.loader;

import com.quasarbyte.llama.cpp.jna.library.declaration.ggml.GgmlLibrary;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.reader.GgmlBackendPathReaderImpl;
import com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.validator.GgmlBackendPathValidatorImpl;

public class GgmlBackendLoaderFactory {
    public GgmlBackendLoader create(GgmlLibrary ggmlLibrary) {
        return new GgmlBackendLoaderImpl(ggmlLibrary, new GgmlBackendPathReaderImpl(), new GgmlBackendPathValidatorImpl());
    }
}
