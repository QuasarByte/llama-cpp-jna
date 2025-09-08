package com.quasarbyte.llama.cpp.jna.binding.llama.processing;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaProcessingBindingFactory {

    public LlamaProcessingBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaProcessingBindingImpl(llamaLibrary);
    }
}