package com.quasarbyte.llama.cpp.jna.model.library.ggml;

import com.sun.jna.Pointer;
import com.sun.jna.PointerType;

/**
 * Pointer wrapper for {@code ggml_backend_buffer_type_t}.
 */
public class GgmlBackendBufferTypeNative extends PointerType {

    public GgmlBackendBufferTypeNative() {
        super();
    }

    public GgmlBackendBufferTypeNative(Pointer pointer) {
        super(pointer);
    }
}
