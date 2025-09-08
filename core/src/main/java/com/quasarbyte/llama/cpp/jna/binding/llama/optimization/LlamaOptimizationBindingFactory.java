package com.quasarbyte.llama.cpp.jna.binding.llama.optimization;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaOptimizationBindingFactory {

    public LlamaOptimizationBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaOptimizationBindingImpl(llamaLibrary);
    }
}