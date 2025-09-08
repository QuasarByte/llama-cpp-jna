package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Collections;
import java.util.List;

/**
 * Native structure for llama sampler chain parameters.
 * <p>
 * This corresponds to the {@code llama_sampler_chain_params} structure from llama.h.
 * Used for configuring sampler chain creation with {@code llama_sampler_chain_init()}.
 * <p>
 * <strong>Performance Monitoring:</strong> Controls whether sampling performance
 * metrics are collected during inference operations.
 * <p>
 * <strong>Usage:</strong> Typically used with sampler chain management functions
 * to configure timing measurement behavior for debugging and optimization.
 */
public class LlamaSamplerChainParamsNative extends Structure implements Structure.ByValue {
    /**
     * Disable performance timing measurements for sampling operations.
     * <p>
     * When set to 0 (false), performance timings are collected during sampling
     * operations, which can be useful for debugging and optimization.
     * When set to 1 (true), timing measurements are disabled to reduce overhead.
     * <p>
     * <strong>Performance Impact:</strong> Disabling measurements can provide
     * slight performance improvements in production environments where timing
     * data is not needed.
     * <p>
     * <strong>Default:</strong> Enabled (0) - measurements are collected
     */
    public byte no_perf = 0;

    /**
     * Defines the order of fields in the native structure.
     * <p>
     * This order must match exactly with the C structure definition
     * to ensure proper memory layout and JNA marshalling.
     */
    @Override
    protected List<String> getFieldOrder() {
        return Collections.singletonList("no_perf");
    }
    
    /**
     * Creates a new LlamaSamplerChainParamsNative instance with default values.
     * <p>
     * Default configuration:
     * <ul>
     *   <li>no_perf = 0 (performance measurements enabled)</li>
     * </ul>
     */
    public LlamaSamplerChainParamsNative() {
        super();
    }
    
    /**
     * Creates a new LlamaSamplerChainParamsNative instance from a native pointer.
     * <p>
     * This constructor is used when the structure is returned from
     * native functions or when working with existing native memory.
     * The structure is automatically read from the native memory.
     * 
     * @param peer pointer to existing native structure
     */
    public LlamaSamplerChainParamsNative(Pointer peer) {
        super(peer);
        read();
    }
}
