package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * Native mapping for {@code llama_opt_params}.
 */
@Structure.FieldOrder({"n_ctx_train", "param_filter", "param_filter_ud", "get_opt_pars", "get_opt_pars_ud", "optimizer_type"})
public class LlamaOptParamsNative extends Structure implements Structure.ByValue {

    public int n_ctx_train;
    public Pointer param_filter;
    public Pointer param_filter_ud;
    public Pointer get_opt_pars;
    public Pointer get_opt_pars_ud;
    public int optimizer_type;

    public LlamaOptParamsNative() {
        super();
    }

    public LlamaOptParamsNative(Pointer pointer) {
        super(pointer);
        read();
    }
}
