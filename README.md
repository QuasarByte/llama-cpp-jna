# llama-cpp-jna

Java Native Access (JNA) wrapper for [llama.cpp](https://github.com/ggml-org/llama.cpp), providing Java bindings to run Large Language Models locally with high performance.

## Features

- **Direct JNA bindings** to llama.cpp native libraries
- **Multi-module Maven structure** with Java 8 compatibility
- **CUDA acceleration support** for GPU inference
- **Cross-platform compatibility** (Windows, Linux, macOS)
- **High-level and low-level API** options for different use cases
- **Example implementations** including SimpleChat interactive demo

## Quick Start

### Prerequisites

- **JDK 25+** - Download from https://jdk.java.net/25/
- **Maven 3.6+** - For building the project
- **Git** - For cloning the repository

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/llama-cpp-jna.git
   cd llama-cpp-jna
   ```

2. **Download llama.cpp binaries** from https://github.com/ggml-org/llama.cpp/releases/tag/b6527

3. **Setup binaries** (see [Binary Setup](#binary-setup) section below)

4. **Download a model** (see [Model Setup](#model-setup) section below)

5. **Run the example**:
   ```bash
   run-simple-chat.cmd          # Windows
   ./run-simple-chat.sh         # Linux/macOS (coming soon)
   ```

## Binary Setup

### Basic Setup (CPU Only)
Extract the llama.cpp binaries to `C:\opt\llama.cpp-b6527-bin` (Windows) or `/opt/llama.cpp-b6527-bin` (Linux/macOS).

### CUDA Setup (GPU Acceleration)
For CUDA acceleration support, you need files from **both** archives:

1. **Download and extract** `llama-b6527-bin-win-cuda-12.4-x64.zip` to `C:\opt\llama.cpp-b6527-bin\`
2. **Download and extract** `cudart-llama-bin-win-cuda-12.4-x64.zip` and copy these CUDA runtime files to the same directory:
   - `cublas64_12.dll`
   - `cublasLt64_12.dll`
   - `cudart64_12.dll`

**Important**: Both archives must be extracted to the same directory for CUDA compatibility.

## Model Setup

### Download Models
Visit the [GGML Models collection](https://huggingface.co/ggml-org/collections) for available models.

**Example - Qwen3 8B Model**:
1. Go to https://huggingface.co/ggml-org/Qwen3-8B-GGUF
2. Download `Qwen3-8B-Q8_0.gguf`
3. Save to `C:\opt\models\Qwen3-8B-Q8_0.gguf` (Windows) or `/opt/models/Qwen3-8B-Q8_0.gguf` (Linux/macOS)


## Running Examples

### Command Line (Recommended)

**Windows:**
```bash
# Option 1: Direct execution (compiles and runs)
run-simple-chat.cmd

# Option 2: Using Maven
run-simple-chat-with-maven.cmd
```

**Linux/macOS:**
```bash
# Coming soon - bash scripts in development
./run-simple-chat.sh
```

### IDE Setup (IntelliJ IDEA)

1. **Configure environment variables** in `llama-cpp-bin.env`:
   ```
   PATH=%PATH%;C:\opt\llama.cpp-b6527-bin
   GGML_BACKEND_PATH=C:\opt\llama.cpp-b6527-bin
   ```

2. **Create run configuration**:
   - **Name**: `SimpleChat`
   - **Main class**: `com.quasarbyte.llama.cpp.jna.examples.simplechat.SimpleChat`
   - **Module**: `examples`
   - **Program arguments**: `-m C:\opt\models\Qwen3-8B-Q8_0.gguf -c 32768 -ngl 100`
   - **Working directory**: Project root
   - **Environment variables**: Import from `llama-cpp-bin.env`

### Command Line Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `-m` | Path to GGUF model file | `-m C:\opt\models\Qwen3-8B-Q8_0.gguf` |
| `-c` | Context length (tokens) | `-c 32768` |
| `-ngl` | GPU layers (0 for CPU-only) | `-ngl 100` |

## Project Structure

```
llama-cpp-jna/
├── core/                           # Main JNA library bindings
│   └── src/main/java/com/quasarbyte/llama/cpp/jna/
│       ├── library/declaration/    # Native library interfaces
│       │   ├── llama/             # Core llama.cpp bindings
│       │   ├── ggml/              # GGML backend bindings
│       │   └── cuda/              # CUDA acceleration bindings
│       ├── bindings/              # High-level bindings layer
│       └── model/                 # Data models and DTOs
├── examples/                       # Usage examples
│   └── src/main/java/com/quasarbyte/llama/cpp/jna/examples/
│       ├── simple/                # Basic usage
│       ├── simplechat/            # Interactive chat
│       └── cuda/                  # CUDA utilities
├── run-simple-chat.cmd            # Windows execution script
├── run-simple-chat-with-maven.cmd # Windows Maven execution
└── llama-cpp-bin.env             # Environment configuration
```

## Building from Source

```bash
# Full build with tests
mvn clean install

# Quick build (skip tests)
mvn clean install -DskipTests

# Build specific module
mvn clean install -pl core

# Copy dependencies for examples
mvn dependency:copy-dependencies -DoutputDirectory=examples/target/lib -pl examples
```

## Windows Compatibility Notes

The prebuilt Windows binaries for `llama.cpp` (build `b6527`) are linked against the latest Microsoft Visual C++ Redistributable. When launching through the JVM, the Java distribution may bring its own MSVC runtime copy:

- **JDK 25+**: Ships compatible DLLs that work without changes
- **JDK 8–24**: Bundle older runtime versions that can cause native loading errors

### Troubleshooting Runtime Issues

If using JDK 8–24, either:
1. **Upgrade to JDK 25+** (recommended)
2. **Remove/rename** bundled MSVC runtime DLLs from `<java.home>/bin`
3. **Ensure** matching Visual C++ Redistributable is installed globally

**Common failure pattern**:
```
llama.dll
├── ggml-cuda.dll
│   ├── cudart64_12.dll, nvcuda.dll, cublas64_12.dll, cublasLt64_12.dll
│   ├── vcruntime140.dll   (from JDK bin - causes conflict)
│   └── msvcp140.dll       (from JDK bin - causes conflict)
```

### Helpful Links
- [Visual C++ Redistributable Downloads](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- [Redistributing Visual C++ Files](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files?view=msvc-170)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/QuasarByte/llama-cpp-jna/issues)
- **Discussions**: Join the community on [GitHub Discussions](https://github.com/QuasarByte/llama-cpp-jna/discussions)
- **Email**: [hello@quasarbyte.com](mailto:hello@quasarbyte.com?subject=Question%20about%20llama-cpp-jna)
- **LinkedIn**: [Connect with the author](https://www.linkedin.com/in/taluyev/)
- **Business inquiries**: https://quasarbyte.com/contact.php