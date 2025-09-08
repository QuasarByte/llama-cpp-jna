package com.quasarbyte.llama.cpp.jna.binding.llama.context;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaContextBindingFactory {

    public LlamaContextBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaContextBindingImpl(llamaLibrary);
    }

}
