package com.quasarbyte.llama.cpp.jna.model.library;

public class LlamaAdapterServiceSettings {

    private Integer adapterMetadataBufferSize;
    private Integer adapterMetadataKeyBufferSize;
    private Integer adapterMetadataValueBufferSize;

    public Integer getAdapterMetadataBufferSize() {
        return adapterMetadataBufferSize;
    }

    public LlamaAdapterServiceSettings setAdapterMetadataBufferSize(Integer adapterMetadataBufferSize) {
        this.adapterMetadataBufferSize = adapterMetadataBufferSize;
        return this;
    }

    public Integer getAdapterMetadataKeyBufferSize() {
        return adapterMetadataKeyBufferSize;
    }

    public LlamaAdapterServiceSettings setAdapterMetadataKeyBufferSize(Integer adapterMetadataKeyBufferSize) {
        this.adapterMetadataKeyBufferSize = adapterMetadataKeyBufferSize;
        return this;
    }

    public Integer getAdapterMetadataValueBufferSize() {
        return adapterMetadataValueBufferSize;
    }

    public LlamaAdapterServiceSettings setAdapterMetadataValueBufferSize(Integer adapterMetadataValueBufferSize) {
        this.adapterMetadataValueBufferSize = adapterMetadataValueBufferSize;
        return this;
    }
}
