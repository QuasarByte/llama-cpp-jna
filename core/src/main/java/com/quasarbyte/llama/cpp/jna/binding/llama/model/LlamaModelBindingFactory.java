package com.quasarbyte.llama.cpp.jna.binding.llama.model;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReaderFactory;

public class LlamaModelBindingFactory {

    public LlamaModelBinding create(LlamaLibrary llamaLibrary) {
        LlamaStringBufferReader llamaStringBufferReader = new LlamaStringBufferReaderFactory().create();
        return new LlamaModelBindingImpl(llamaLibrary, llamaStringBufferReader);
    }

}
