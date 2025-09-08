package com.quasarbyte.llama.cpp.jna.binding.stringbuffer;

public class LlamaStringBufferReaderFactory {

    public LlamaStringBufferReader create() {
        return new LlamaStringBufferReaderImpl();
    }

}
