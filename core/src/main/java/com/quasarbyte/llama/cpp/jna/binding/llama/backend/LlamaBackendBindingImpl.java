package com.quasarbyte.llama.cpp.jna.binding.llama.backend;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaBackendBindingImpl implements LlamaBackendBinding {

    public final LlamaLibrary llamaLibrary;

    public LlamaBackendBindingImpl(LlamaLibrary llamaLibrary) {
        this.llamaLibrary = llamaLibrary;
    }

    @Override
    public void backendInit() {
        llamaLibrary.llama_backend_init();
    }

    @Override
    public void freeBackend() {
        llamaLibrary.llama_backend_free();
    }

}
