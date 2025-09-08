package com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.reader;

public class GgmlBackendPathReaderFactory {

    public static GgmlBackendPathReader create(){
        return new GgmlBackendPathReaderImpl();
    }

}
