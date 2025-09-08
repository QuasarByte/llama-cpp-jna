package com.quasarbyte.llama.cpp.jna.library.declaration.llama;

import com.quasarbyte.llama.cpp.jna.library.declaration.UInt32;

public interface LlamaConstants {
    // Default random seed (32-bit unsigned -> stored in long)
    UInt32 LLAMA_DEFAULT_SEED = new UInt32(0xFFFFFFFFL);

    // Null token indicator
    int LLAMA_TOKEN_NULL = -1;

    // File magic numbers
    int LLAMA_FILE_MAGIC_GGLA = 0x67676C61; // 'ggla'
    int LLAMA_FILE_MAGIC_GGSN = 0x6767736E; // 'ggsn'
    int LLAMA_FILE_MAGIC_GGSQ = 0x67677371; // 'ggsq'

    // Session constants
    int LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN;
    int LLAMA_SESSION_VERSION = 9;

    // State sequence constants
    int LLAMA_STATE_SEQ_MAGIC = LLAMA_FILE_MAGIC_GGSQ;
    int LLAMA_STATE_SEQ_VERSION = 2;
}
