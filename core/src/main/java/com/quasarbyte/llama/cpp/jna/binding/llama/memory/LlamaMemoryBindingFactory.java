package com.quasarbyte.llama.cpp.jna.binding.llama.memory;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaMemoryBindingFactory {

    public LlamaMemoryBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaMemoryBindingImpl(llamaLibrary);
    }
}