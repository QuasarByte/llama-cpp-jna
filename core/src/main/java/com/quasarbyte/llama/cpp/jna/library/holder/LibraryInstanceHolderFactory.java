package com.quasarbyte.llama.cpp.jna.library.holder;

import com.sun.jna.Library;

import java.util.Map;

/**
 * Simple factory for creating LibraryInstanceHolder instances.
 * <p>
 * Keeps the creation logic in one place, hides the concrete implementation,
 * and allows you to change the implementation later without affecting clients.
 */
public class LibraryInstanceHolderFactory {

    private LibraryInstanceHolderFactory() {
        // prevent instantiation
    }

    public static <T extends Library> LibraryInstanceHolder<T> create(
            String libraryName,
            Map<String, Object> libraryOptions,
            Class<T> libraryClass) {
        return new LibraryInstanceHolderImpl<>(libraryName, libraryOptions, libraryClass);
    }
}
