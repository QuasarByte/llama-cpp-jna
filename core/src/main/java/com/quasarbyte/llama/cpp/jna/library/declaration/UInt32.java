package com.quasarbyte.llama.cpp.jna.library.declaration;

/**
 * uint32_t
 */
public class UInt32 extends com.sun.jna.IntegerType {
    public UInt32() {
        super(4, true);
    }       // 4 bytes, unsigned

    public UInt32(long v) {
        super(4, v, true);
    }
}
