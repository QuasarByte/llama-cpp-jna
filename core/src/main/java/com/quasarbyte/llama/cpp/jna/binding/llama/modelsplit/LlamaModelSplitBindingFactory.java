package com.quasarbyte.llama.cpp.jna.binding.llama.modelsplit;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReaderFactory;

public class LlamaModelSplitBindingFactory {

    public LlamaModelSplitBinding create(LlamaLibrary llamaLibrary) {
        LlamaStringBufferReader llamaStringBufferReader = new LlamaStringBufferReaderFactory().create();
        return new LlamaModelSplitBindingImpl(llamaLibrary, llamaStringBufferReader);
    }
}