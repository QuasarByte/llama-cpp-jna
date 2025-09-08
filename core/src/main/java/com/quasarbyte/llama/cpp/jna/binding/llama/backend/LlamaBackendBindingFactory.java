package com.quasarbyte.llama.cpp.jna.binding.llama.backend;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaBackendBindingFactory {
    public LlamaBackendBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaBackendBindingImpl(llamaLibrary);
    }
}
