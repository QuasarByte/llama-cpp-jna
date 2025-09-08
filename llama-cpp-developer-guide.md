# llama.cpp Developer Guide

This guide provides a comprehensive overview of the `llama.cpp` library and how to use it to develop custom applications.

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Getting Started](#2-getting-started)
  - [2.1. Initialization](#21-initialization)
  - [2.2. Model Loading](#22-model-loading)
  - [2.3. Context Creation](#23-context-creation)
  - [2.4. Tokenization](#24-tokenization)
  - [2.5. Inference](#25-inference)
  - [2.6. Sampling](#26-sampling)
  - [2.7. Generating Text](#27-generating-text)
- [3. Advanced Topics](#3-advanced-topics)
  - [3.1. Batching](#31-batching)
  - [3.2. Parallel Processing](#32-parallel-processing)
  - [3.3. Embeddings](#33-embeddings)
  - [3.4. Saving and Loading State](#34-saving-and-loading-state)
  - [3.5. Advanced KV Cache Management](#35-advanced-kv-cache-management)
- [4. Chat Applications](#4-chat-applications)
- [5. Backends](#5-backends)
  - [5.1. CPU](#51-cpu)
  - [5.2. GPU](#52-gpu)
- [6. Error Handling](#6-error-handling)
- [7. Performance Tips](#7-performance-tips)
- [8. Conclusion](#8-conclusion)

## 1. Introduction

`llama.cpp` is a C/C++ library for running Large Language Models (LLMs) locally. It is designed for high performance and portability, with support for various hardware backends (CPU, GPU) and operating systems.

This guide will walk you through the key concepts and APIs of the library, with code examples to help you get started.

## 2. Getting Started

To use `llama.cpp` in your own project, you need to include the `llama.h` header file and link against the `llama` library.

The best way to understand how to use the library is to look at the examples provided in the `examples` directory. The `simple` example is a good starting point.

### 2.1. Initialization

Before you can use the library, you need to initialize the backend.

```cpp
#include "llama.h"

int main() {
    llama_backend_init();
    // ...
    llama_backend_free();
    return 0;
}
```

The `llama_backend_init()` function initializes the backend. You can control the NUMA (Non-Uniform Memory Access) behavior with `llama_numa_init()`. The `llama_backend_free()` function frees the resources used by the backend.

### 2.2. Model Loading

To load a model, you need to use the `llama_model_load_from_file()` function. This function takes the path to the model file and a `llama_model_params` struct as input.

```cpp
lama_model_params model_params = llama_model_default_params();
model_params.n_gpu_layers = 99; // Offload all layers to GPU

lama_model * model = llama_model_load_from_file("path/to/model.gguf", model_params);
if (!model) {
    fprintf(stderr, "error: unable to load model\n");
    return 1;
}
```

The `llama_model_default_params()` function returns a `llama_model_params` struct with default values. You can modify this struct to customize the model loading process. For example, you can set the number of GPU layers to offload to the GPU.

### 2.3. Context Creation

Once you have loaded a model, you need to create a context. The context holds the state of the model and is used for inference.

```cpp
lama_context_params ctx_params = llama_context_default_params();
ctx_params.n_ctx = 2048; // Context size

lama_context * ctx = llama_init_from_model(model, ctx_params);
if (!ctx) {
    fprintf(stderr, "error: failed to create the llama_context\n");
    return 1;
}
```

The `llama_context_default_params()` function returns a `llama_context_params` struct with default values. You can modify this struct to customize the context creation process. For example, you can set the context size (`n_ctx`).

### 2.4. Tokenization

Before you can run inference, you need to tokenize the input prompt. The `llama_tokenize()` function can be used for this purpose.

```cpp
const llama_vocab * vocab = llama_model_get_vocab(model);

std::vector<llama_token> tokens;
tokens = common_tokenize(vocab, "Hello, world!", true);
```

The `common_tokenize` function is a helper function from `common.h` that simplifies tokenization.

### 2.5. Inference

To run inference, you need to use the `llama_decode()` function. This function takes a `llama_batch` as input.

```cpp
lama_batch batch = llama_batch_init(tokens.size(), 0, 1);
for (size_t i = 0; i < tokens.size(); i++) {
    common_batch_add(batch, tokens[i], i, {0}, false);
}
batch.logits[batch.n_tokens - 1] = true; // We want to get the logits for the last token

if (llama_decode(ctx, batch) != 0) {
    fprintf(stderr, "llama_decode() failed\n");
    return 1;
}
```

The `llama_batch_init()` function initializes a `llama_batch`. The `common_batch_add()` function is a helper to add a token to the batch. The `logits` field of the batch determines for which tokens the logits will be returned.

### 2.6. Sampling

After running inference, you can use a sampler to sample the next token from the logits.

```cpp
lama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
lama_sampler_chain_add(smpl, llama_sampler_init_greedy());

const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
```

The `llama_sampler_chain_init()` function initializes a sampler chain. You can add different samplers to the chain, such as greedy sampling, top-k sampling, and top-p sampling. The `llama_sampler_sample()` function samples the next token.

### 2.7. Generating Text

By repeatedly running inference and sampling, you can generate text.

```cpp
for (int i = 0; i < n_predict; ++i) {
    const llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

    if (llama_vocab_is_eog(vocab, new_token_id)) {
        break;
    }

    printf("%s", common_token_to_piece(ctx, new_token_id).c_str());

    common_batch_clear(batch);
    common_batch_add(batch, new_token_id, n_past + i, {0}, true);

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "failed to decode\n");
        return 1;
    }
}
```

## 3. Advanced Topics

### 3.1. Batching

`llama.cpp` supports batching to improve performance. The `batched` example shows how to use batching to process multiple sequences in parallel.

The key idea is to create a `llama_batch` with a larger size and add tokens from different sequences to it. The `llama_decode()` function will then process the entire batch in a single call.

### 3.2. Parallel Processing

The `parallel` example demonstrates how to use `llama.cpp` to simulate a server with multiple clients. Each client has its own sequence, and the server processes the requests in parallel.

This example uses a single context and a single batch, but it uses different sequence IDs to distinguish between the clients.

### 3.3. Embeddings

The `embedding` example shows how to use `llama.cpp` to generate embeddings for a given text.

To get embeddings, you need to set the `embedding` parameter to `true` in the `common_params`. The `llama_get_embeddings()` function can then be used to get the embeddings.

### 3.4. Saving and Loading State

The `save-load-state` example shows how to save and load the state of the model and context.

The `llama_state_get_size()` and `llama_state_get_data()` functions can be used to get the state data. The `llama_state_set_data()` function can be used to set the state data.

This is useful for saving the state of a long-running generation and resuming it later.

### 3.5. Advanced KV Cache Management

The KV cache stores the key-value pairs for the tokens that have been processed. This allows the model to reuse the computations for the tokens that have already been seen.

`llama.cpp` provides several functions for managing the KV cache:

*   `llama_kv_cache_seq_cp()`: Copies the KV cache from one sequence to another.
*   `llama_kv_cache_seq_rm()`: Removes a range of tokens from the KV cache for a specific sequence.
*   `llama_kv_cache_seq_div()`: Divides the KV cache for a specific sequence.

These functions can be used to implement more advanced generation strategies, such as speculative decoding and tree-based search.

The `parallel` example shows how to use `llama_kv_cache_seq_cp` to share the system prompt's KV cache among multiple sequences.

## 4. Chat Applications

The `simple-chat` example shows how to build a simple chat application using `llama.cpp`.

This example uses a loop to get input from the user, generate a response, and then print the response. It also shows how to use a chat template to format the conversation.

## 5. Backends

`llama.cpp` supports different backends for running inference on different hardware.

### 5.1. CPU

The CPU backend is the default backend. It is optimized for performance on a wide range of CPUs.

### 5.2. GPU

`llama.cpp` supports NVIDIA GPUs via CUDA and AMD GPUs via HIP. To use the GPU backend, you need to set the `n_gpu_layers` parameter in the `llama_model_params` to a value greater than 0.

The `build.md` file in the `docs` directory provides instructions on how to build `llama.cpp` with GPU support.

## 6. Error Handling

It is important to handle errors when using `llama.cpp`. Most functions in the library return an error code if something goes wrong.

For example, the `llama_model_load_from_file()` function returns a `nullptr` if it fails to load the model. The `llama_decode()` function returns a non-zero value if it fails to decode the batch.

```cpp
if (llama_decode(ctx, batch) != 0) {
    fprintf(stderr, "llama_decode() failed\n");
    return 1;
}
```

You should always check the return values of the functions you call and handle errors appropriately.

## 7. Performance Tips

Here are some tips for optimizing the performance of `llama.cpp`:

*   **Choose the right parameters:** The performance of `llama.cpp` can be affected by the parameters you choose. For example, the `n_ctx` and `n_batch` parameters can affect the memory usage and the inference speed. You should experiment with different parameters to find the best values for your application.
*   **Use batching effectively:** Batching can significantly improve the performance of `llama.cpp`. You should try to process as many tokens as possible in a single batch. The `batched` and `parallel` examples show how to use batching effectively.
*   **Use the GPU backend:** If you have a supported GPU, you can use the GPU backend to accelerate inference. To use the GPU backend, you need to set the `n_gpu_layers` parameter in the `llama_model_params` to a value greater than 0.
*   **Use the right build options:** When building `llama.cpp`, you can enable different build options to optimize for performance. For example, you can enable `LLAMA_CUBLAS` to use cuBLAS for matrix multiplication on NVIDIA GPUs.

## 8. Conclusion

This guide has provided a comprehensive overview of the `llama.cpp` library. By following the examples and the documentation, you should be able to use `llama.cpp` to develop your own custom applications.

```