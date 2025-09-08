package com.quasarbyte.llama.cpp.jna.binding.llama.sampler;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;

public class LlamaSamplerBindingFactory {

    public LlamaSamplerBinding create(LlamaLibrary llamaLibrary) {
        return new LlamaSamplerBindingImpl(llamaLibrary);
    }

}
