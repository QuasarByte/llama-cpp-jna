package com.quasarbyte.llama.cpp.jna.binding.llama.adapter;

import com.quasarbyte.llama.cpp.jna.library.declaration.llama.LlamaLibrary;
import com.quasarbyte.llama.cpp.jna.model.library.LlamaAdapterServiceSettings;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReader;
import com.quasarbyte.llama.cpp.jna.binding.stringbuffer.LlamaStringBufferReaderFactory;

import static com.quasarbyte.llama.cpp.jna.model.library.LlamaAdapterServiceConstants.*;

public class LlamaAdapterBindingFactory {

    public LlamaAdapterBinding create(LlamaLibrary llamaLibrary) {
        LlamaStringBufferReader llamaStringBufferReader = new LlamaStringBufferReaderFactory().create();
        LlamaAdapterServiceSettings llamaAdapterServiceSettings = new LlamaAdapterServiceSettings()
                .setAdapterMetadataBufferSize(GET_ADAPTER_METADATA_BUFFER_SIZE)
                .setAdapterMetadataKeyBufferSize(GET_ADAPTER_METADATA_KEY_BUFFER_SIZE)
                .setAdapterMetadataValueBufferSize(GET_ADAPTER_METADATA_VALUE_BUFFER_SIZE);
        return new LlamaAdapterBindingImpl(llamaLibrary, llamaStringBufferReader, llamaAdapterServiceSettings);
    }
}