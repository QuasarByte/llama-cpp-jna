package com.quasarbyte.llama.cpp.jna.binding.llama.result;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaResultBindingFactory {

    public LlamaResultBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaResultBindingImpl(llamaLibrary);
    }
}