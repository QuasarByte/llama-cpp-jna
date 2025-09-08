package com.quasarbyte.llama.cpp.jna.binding.llama.token;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReaderFactory;

public class LlamaTokenBindingFactory {

    public LlamaTokenBinding create(LlamaLibrary llamaLibrary) {
        LlamaStringBufferReader llamaStringBufferReader = new LlamaStringBufferReaderFactory().create();
        return new LlamaTokenBindingImpl(llamaLibrary,  llamaStringBufferReader);
    }

}
