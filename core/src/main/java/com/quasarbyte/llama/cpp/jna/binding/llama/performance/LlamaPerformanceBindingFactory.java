package com.quasarbyte.llama.cpp.jna.binding.llama.performance;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaPerformanceBindingFactory {

    public LlamaPerformanceBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaPerformanceBindingImpl(llamaLibrary);
    }
}