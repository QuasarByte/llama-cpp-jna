package com.quasarbyte.llama.cpp.jna.model.cuda;

public class CudaDevice {

    private Integer id;
    private String description;

    public Integer getId() {
        return id;
    }

    public CudaDevice setId(Integer id) {
        this.id = id;
        return this;
    }

    public String getDescription() {
        return description;
    }

    public CudaDevice setDescription(String description) {
        this.description = description;
        return this;
    }
}
