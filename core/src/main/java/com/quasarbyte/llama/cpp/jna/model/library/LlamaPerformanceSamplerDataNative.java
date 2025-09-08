package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * JNA mapping for the native llama_perf_sampler_data structure.
 * <p>
 * struct llama_perf_sampler_data {
 *     double t_sample_ms;
 * <p>
 *     int32_t n_sample;
 * };
 */
@Structure.FieldOrder({"t_sample_ms", "n_sample"})
public class LlamaPerformanceSamplerDataNative extends Structure {

    public double t_sample_ms;
    public int n_sample;

    public LlamaPerformanceSamplerDataNative() {
        super();
    }

    public LlamaPerformanceSamplerDataNative(Pointer pointer) {
        super(pointer);
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("t_sample_ms", "n_sample");
    }

    public static class ByReference extends LlamaPerformanceSamplerDataNative implements Structure.ByReference {
        public ByReference() {}
        public ByReference(Pointer pointer) { super(pointer); }
    }

    public static class ByValue extends LlamaPerformanceSamplerDataNative implements Structure.ByValue {
        public ByValue() {}
    }
}