package com.quasarbyte.llama.cpp.jna.binding.llama.chat;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReaderFactory;

public class LlamaChatBindingFactory {

    public LlamaChatBinding create(LlamaLibrary llamaLibrary) {
        LlamaStringBufferReader llamaStringBufferReader = new LlamaStringBufferReaderFactory().create();
        return new LlamaChatBindingImpl(llamaLibrary, llamaStringBufferReader);
    }
}