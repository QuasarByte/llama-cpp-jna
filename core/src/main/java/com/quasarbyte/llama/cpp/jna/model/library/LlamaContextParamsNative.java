package com.quasarbyte.llama.cpp.jna.model.library;

import com.sun.jna.Pointer;
import com.sun.jna.Structure;

/**
 * Native structure for llama context parameters.
 * <p>
 * This corresponds to the {@code llama_context_params} structure from llama.h.
 * Used for configuring context creation with {@code llama_init_from_model()}.
 * <p>
 * <strong>Important:</strong> Changing the default values of parameters marked as
 * [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations.
 * 
 * @see <a href="https://github.com/ggml-org/llama.cpp/pull/7544">Experimental Parameters Issue</a>
 */
public class LlamaContextParamsNative extends Structure implements Structure.ByValue {
    /** Text context size (0 = use value from model) */
    public int n_ctx = 0;
    
    /** Logical maximum batch size that can be submitted to llama_decode */
    public int n_batch = 0;
    
    /** Physical maximum batch size */
    public int n_ubatch = 0;
    
    /** Maximum number of sequences (i.e. distinct states for recurrent models) */
    public int n_seq_max = 0;
    
    /** Number of threads to use for generation (-1 = auto) */
    public int n_threads = -1;
    
    /** Number of threads to use for batch processing (-1 = auto) */
    public int n_threads_batch = -1;

    /** RoPE scaling type (from LlamaRopeScalingType enum) */
    public int rope_scaling_type = 0;
    
    /** Pooling type for embeddings (from LlamaPoolingType enum) */
    public int pooling_type = 0;
    
    /** Attention type to use for embeddings (from LlamaAttentionType enum) */
    public int attention_type = 0;
    
    /** Flash Attention configuration (from LlamaFlashAttnType enum) */
    public int flash_attn_type = 0;

    //
    // RoPE and YaRN Configuration
    // @see <a href="https://github.com/ggml-org/llama.cpp/pull/2054">RoPE Scaling PR</a>
    //
    
    /** RoPE base frequency (0 = use value from model) */
    public float rope_freq_base = 0.0f;
    
    /** RoPE frequency scaling factor (0 = use value from model) */
    public float rope_freq_scale = 0.0f;
    
    /** YaRN extrapolation mix factor (negative = use value from model) */
    public float yarn_ext_factor = 0.0f;
    
    /** YaRN magnitude scaling factor */
    public float yarn_attn_factor = 0.0f;
    
    /** YaRN low correction dimension */
    public float yarn_beta_fast = 0.0f;
    
    /** YaRN high correction dimension */
    public float yarn_beta_slow = 0.0f;
    
    /** YaRN original context size */
    public int yarn_orig_ctx = 0;
    
    /** @deprecated Defragment the KV cache if holes/size > threshold, <= 0 disabled */
    @Deprecated
    public float defrag_thold = 0.0f;

    //
    // Callback Functions
    //
    
    /** Backend scheduler evaluation callback (ggml_backend_sched_eval_callback) */
    public Pointer cb_eval = null;
    
    /** User data passed to evaluation callback */
    public Pointer cb_eval_user_data = null;

    //
    // Cache Configuration [EXPERIMENTAL]
    //
    
    /** Data type for K cache [EXPERIMENTAL] (ggml_type enum) */
    public int type_k = 0;
    
    /** Data type for V cache [EXPERIMENTAL] (ggml_type enum) */
    public int type_v = 0;

    //
    // Abort Callback
    // If the callback returns true, execution of llama_decode() will be aborted.
    // Currently works only with CPU execution.
    //
    
    /** Abort callback function (ggml_abort_callback) */
    public Pointer abort_callback = null;
    
    /** User data passed to abort callback */
    public Pointer abort_callback_data = null;

    //
    // Boolean Flags
    // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
    //
    
    /** If true, extract embeddings (together with logits) */
    public byte embeddings = 0;
    
    /** Offload the KQV ops (including the KV cache) to GPU */
    public byte offload_kqv = 0;
    
    /** Measure performance timings (false to enable, true to disable) */
    public byte no_perf = 0;
    
    /** Offload host tensor operations to device */
    public byte op_offload = 0;
    
    /** 
     * Use full-size SWA cache.
     * <p>
     * <strong>Performance Note:</strong> Setting to false when n_seq_max > 1 
     * can cause bad performance in some cases.
     * @see <a href="https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055">SWA Cache Issue</a>
     * @see <a href="https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573">Performance Reference</a>
     */
    public byte swa_full = 0;
    
    /** 
     * Use a unified buffer across input sequences when computing attention.
     * <p>
     * Try to disable when n_seq_max > 1 for improved performance when the sequences 
     * do not share a large prefix.
     * @see <a href="https://github.com/ggml-org/llama.cpp/pull/14363">Unified Buffer PR</a>
     */
    public byte kv_unified = 0;

    @Override
    protected java.util.List<String> getFieldOrder() {
        return java.util.Arrays.asList("n_ctx", "n_batch", "n_ubatch", "n_seq_max", "n_threads", "n_threads_batch",
                "rope_scaling_type", "pooling_type", "attention_type", "flash_attn_type",
                "rope_freq_base", "rope_freq_scale", "yarn_ext_factor", "yarn_attn_factor",
                "yarn_beta_fast", "yarn_beta_slow", "yarn_orig_ctx", "defrag_thold",
                "cb_eval", "cb_eval_user_data", "type_k", "type_v",
                "abort_callback", "abort_callback_data",
                "embeddings", "offload_kqv", "no_perf", "op_offload", "swa_full", "kv_unified");
    }
}
