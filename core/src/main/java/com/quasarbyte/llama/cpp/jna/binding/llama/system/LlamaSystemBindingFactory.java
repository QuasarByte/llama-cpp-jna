package com.quasarbyte.llama.cpp.jna.binding.llama.system;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaSystemBindingFactory {

    public LlamaSystemBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaSystemBindingImpl(llamaLibrary);
    }
}