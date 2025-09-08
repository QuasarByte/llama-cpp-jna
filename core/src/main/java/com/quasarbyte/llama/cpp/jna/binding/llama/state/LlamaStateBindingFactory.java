package com.quasarbyte.llama.cpp.jna.binding.llama.state;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaStateBindingFactory {

    public LlamaStateBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaStateBindingImpl(llamaLibrary);
    }
}