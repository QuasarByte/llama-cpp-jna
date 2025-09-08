package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * Native structure for llama model quantization parameters.
 * <p>
 * This corresponds to the {@code llama_model_quantize_params} structure from llama.h.
 * Used for configuring model quantization with {@code llama_model_quantize()}.
 * <p>
 * <strong>Quantization:</strong> Controls the quantization format, threading, and
 * specific tensor type overrides for reducing model size and memory usage.
 * <p>
 * <strong>Performance:</strong> Provides control over threading and tensor processing
 * strategies to optimize quantization speed and quality.
 * <p>
 * <strong>Advanced Options:</strong> Supports importance matrix data, layer pruning,
 * and selective tensor quantization for fine-grained control.
 */
public class LlamaModelQuantizeParamsNative extends Structure implements Structure.ByValue {
    /**
     * Number of threads to use for quantizing.
     * <p>
     * If <=0, will use {@code std::thread::hardware_concurrency()} to determine
     * the optimal number of threads based on available CPU cores.
     * <p>
     * <strong>Typical Values:</strong>
     * <ul>
     *   <li>-1 or 0: Auto-detect optimal thread count</li>
     *   <li>1: Single-threaded quantization</li>
     *   <li>2-N: Specific thread count for parallel processing</li>
     * </ul>
     */
    public int nthread = 0;

    /**
     * Target quantization format type.
     * <p>
     * Specifies the quantization format to convert the model to.
     * Uses values from {@code enum llama_ftype}:
     * <ul>
     *   <li>LLAMA_FTYPE_ALL_F32 (0): 32-bit floating point (no quantization)</li>
     *   <li>LLAMA_FTYPE_MOSTLY_F16 (1): 16-bit floating point</li>
     *   <li>LLAMA_FTYPE_MOSTLY_Q4_0 (2): 4-bit quantization</li>
     *   <li>LLAMA_FTYPE_MOSTLY_Q4_1 (3): 4-bit quantization with bias</li>
     *   <li>LLAMA_FTYPE_MOSTLY_Q8_0 (7): 8-bit quantization</li>
     *   <li>And many other K-quantization formats (Q2_K, Q3_K, etc.)</li>
     * </ul>
     */
    public int ftype = 0;

    /**
     * Data type for output tensor quantization.
     * <p>
     * Specifies the quantization format specifically for the output tensor.
     * Uses values from {@code enum ggml_type}. This allows different
     * quantization for the output layer compared to other model layers.
     * Type: {@code enum ggml_type}
     */
    public int output_tensor_type = 0;

    /**
     * Data type for token embedding tensor quantization.
     * <p>
     * Specifies the quantization format specifically for token embedding tensors.
     * Uses values from {@code enum ggml_type}. Token embeddings can be
     * quantized differently from other tensors for optimal performance.
     * Type: {@code enum ggml_type}
     */
    public int token_embedding_type = 0;

    /**
     * Pointer to importance matrix data for guided quantization.
     * <p>
     * Optional importance matrix that guides the quantization process by
     * providing information about which weights are more critical for
     * model performance. Helps preserve model quality during quantization.
     * Type: {@code void*}
     */
    public Pointer imatrix = null;

    /**
     * Pointer to key-value overrides for model metadata.
     * <p>
     * Optional vector containing metadata overrides to apply during quantization.
     * Allows customizing model metadata without modifying the original model.
     * Type: {@code void*} (pointer to vector containing overrides)
     */
    public Pointer kv_overrides = null;

    /**
     * Pointer to tensor type specifications.
     * <p>
     * Optional vector containing specific quantization types for individual tensors.
     * Enables fine-grained control over which tensors use which quantization formats.
     * Type: {@code void*} (pointer to vector containing tensor types)
     */
    public Pointer tensor_types = null;

    /**
     * Pointer to layer indices for pruning.
     * <p>
     * Optional vector containing indices of layers to prune (remove) during
     * quantization. Used for creating smaller, more efficient models by
     * removing less important layers.
     * Type: {@code void*} (pointer to vector containing layer indices)
     */
    public Pointer prune_layers = null;

    //
    // Boolean Flags
    // Keep the booleans together to avoid misalignment during copy-by-value.
    // Use byte instead of boolean to match C bool size exactly.
    //

    /**
     * Allow requantizing non-f32/f16 tensors.
     * <p>
     * When enabled, allows quantization of tensors that are already in
     * quantized formats (not f32 or f16). Useful for converting between
     * different quantization formats.
     * <p>
     * <strong>Default:</strong> Disabled (0)
     */
    public byte allow_requantize = 0;

    /**
     * Quantize the output.weight tensor.
     * <p>
     * Controls whether the final output weight tensor should be quantized.
     * The output tensor often benefits from higher precision, so this
     * can be disabled to maintain f16/f32 precision for better quality.
     * <p>
     * <strong>Default:</strong> Enabled (1)
     */
    public byte quantize_output_tensor = 1;

    /**
     * Only copy tensors without quantization.
     * <p>
     * When enabled, tensors are copied without quantization. The ftype,
     * allow_requantize, and quantize_output_tensor parameters are ignored.
     * Useful for model format conversion without changing precision.
     * <p>
     * <strong>Default:</strong> Disabled (0)
     */
    public byte only_copy = 0;

    /**
     * Quantize all tensors to the default type.
     * <p>
     * When enabled, forces all tensors to use the same quantization format
     * specified by ftype, ignoring any tensor-specific type specifications.
     * Simplifies the quantization process for uniform compression.
     * <p>
     * <strong>Default:</strong> Disabled (0)
     */
    public byte pure = 0;

    /**
     * Maintain the same number of model shards after quantization.
     * <p>
     * When enabled, preserves the original sharding structure of the model.
     * Useful when the original model is split across multiple files and
     * you want to maintain that structure after quantization.
     * <p>
     * <strong>Default:</strong> Disabled (0)
     */
    public byte keep_split = 0;

    /**
     * Defines the order of fields in the native structure.
     * <p>
     * This order must match exactly with the C structure definition
     * to ensure proper memory layout and JNA marshalling.
     */
    @Override
    protected java.util.List<String> getFieldOrder() {
        return java.util.Arrays.asList("nthread", "ftype", "output_tensor_type", "token_embedding_type",
                "imatrix", "kv_overrides", "tensor_types", "prune_layers",
                "allow_requantize", "quantize_output_tensor", "only_copy", "pure", "keep_split");
    }

    /**
     * Creates a new LlamaModelQuantizeParamsNative instance with default values.
     * <p>
     * Default configuration:
     * <ul>
     *   <li>nthread = 0 (auto-detect thread count)</li>
     *   <li>ftype = 0 (LLAMA_FTYPE_ALL_F32 - no quantization)</li>
     *   <li>output_tensor_type = 0 (default type)</li>
     *   <li>token_embedding_type = 0 (default type)</li>
     *   <li>allow_requantize = 0 (disabled)</li>
     *   <li>quantize_output_tensor = 1 (enabled)</li>
     *   <li>only_copy = 0 (disabled)</li>
     *   <li>pure = 0 (disabled)</li>
     *   <li>keep_split = 0 (disabled)</li>
     *   <li>All pointers = null</li>
     * </ul>
     */
    public LlamaModelQuantizeParamsNative() {
        super();
    }

    /**
     * Creates a new LlamaModelQuantizeParamsNative instance from a native pointer.
     * <p>
     * This constructor is used when the structure is returned from
     * native functions or when working with existing native memory.
     * The structure is automatically read from the native memory.
     *
     * @param peer pointer to existing native structure
     */
    public LlamaModelQuantizeParamsNative(Pointer peer) {
        super(peer);
        read();
    }
}