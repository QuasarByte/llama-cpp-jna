package com.quasarbyte.llama.cpp.jna.binding.librarymetadata;

import com.sun.jna.Library;

import java.nio.file.Path;

public interface LibraryMetadataReader {
    Path getLibraryPath(Library proxy);
}
