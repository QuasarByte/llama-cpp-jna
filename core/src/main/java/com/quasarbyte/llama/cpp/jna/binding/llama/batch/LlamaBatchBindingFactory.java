package com.quasarbyte.llama.cpp.jna.binding.llama.batch;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaBatchBindingFactory {

    public LlamaBatchBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaBatchBindingImpl(llamaLibrary);
    }

}
