package com.quasarbyte.llama.cpp.jna.binding.ggml.backend.path.validator;

public class GgmlBackendPathValidatorFactory {

    public static GgmlBackendPathValidator create(){
        return new GgmlBackendPathValidatorImpl();
    }

}
