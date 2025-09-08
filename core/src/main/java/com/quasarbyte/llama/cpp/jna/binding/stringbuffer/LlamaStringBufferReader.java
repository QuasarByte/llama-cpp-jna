package com.quasarbyte.llama.cpp.jna.binding.stringbuffer;

import java.nio.charset.Charset;

public interface LlamaStringBufferReader {

    String readString(byte[] buffer, int length);
    String readString(byte[] buffer, int length, Charset charset);

}
