package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * JNA mapping for the native llama_perf_context_data structure.
 * <p>
 * struct llama_perf_context_data {
 *     double t_start_ms;
 *     double t_load_ms;
 *     double t_p_eval_ms;
 *     double t_eval_ms;
 * <p>
 *     int32_t n_p_eval;
 *     int32_t n_eval;
 *     int32_t n_reused; // number of times a ggml compute graph had been reused
 * };
 */
@Structure.FieldOrder({"t_start_ms", "t_load_ms", "t_p_eval_ms", "t_eval_ms", "n_p_eval", "n_eval", "n_reused"})
public class LlamaPerformanceContextDataNative extends Structure {

    public double t_start_ms;
    public double t_load_ms;
    public double t_p_eval_ms;
    public double t_eval_ms;

    public int n_p_eval;
    public int n_eval;
    public int n_reused;

    public LlamaPerformanceContextDataNative() {
        super();
    }

    public LlamaPerformanceContextDataNative(Pointer pointer) {
        super(pointer);
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("t_start_ms", "t_load_ms", "t_p_eval_ms", "t_eval_ms", "n_p_eval", "n_eval", "n_reused");
    }

    public static class ByReference extends LlamaPerformanceContextDataNative implements Structure.ByReference {
        public ByReference() {}
        public ByReference(Pointer pointer) { super(pointer); }
    }

    public static class ByValue extends LlamaPerformanceContextDataNative implements Structure.ByValue {
        public ByValue() {}
    }

    @Override
    public String toString() {
        return "LlamaPerformanceContextDataNative{" +
                "t_start_ms=" + t_start_ms +
                ", t_load_ms=" + t_load_ms +
                ", t_p_eval_ms=" + t_p_eval_ms +
                ", t_eval_ms=" + t_eval_ms +
                ", n_p_eval=" + n_p_eval +
                ", n_eval=" + n_eval +
                ", n_reused=" + n_reused +
                '}';
    }
}