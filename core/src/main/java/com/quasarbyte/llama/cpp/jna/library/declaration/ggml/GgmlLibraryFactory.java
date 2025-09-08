package com.quasarbyte.llama.cpp.jna.library.declaration.ggml;

import com.quasarbyte.llama.cpp.jna.library.holder.LibraryInstanceHolder;
import com.quasarbyte.llama.cpp.jna.library.holder.LibraryInstanceHolderFactory;
import com.sun.jna.Function;
import com.sun.jna.Library;

import java.util.HashMap;
import java.util.Map;

public class GgmlLibraryFactory {

    private final static String LIBRARY_NAME = "ggml";

    // Use explicit options to force cdecl calling convention
    private final static Map<String, Object> LIBRARY_OPTIONS = new HashMap<String, Object>() {{
        put(Library.OPTION_CALLING_CONVENTION, Function.C_CONVENTION);
    }};

    private final static LibraryInstanceHolder<GgmlLibrary> INSTANCE_HOLDER = LibraryInstanceHolderFactory.create(LIBRARY_NAME, LIBRARY_OPTIONS, GgmlLibrary.class);

    public GgmlLibrary getInstance() {
        return INSTANCE_HOLDER.getInstance();
    }
}
