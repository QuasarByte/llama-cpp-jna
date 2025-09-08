package com.quasarbyte.llama.cpp.jna.binding.librarymetadata;

public class LibraryMetadataReaderFactory {

    public LibraryMetadataReader create() {
        return new LibraryMetadataReaderImpl();
    }

}
