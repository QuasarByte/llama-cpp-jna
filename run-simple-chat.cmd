@echo off

SET JAVA_HOME=C:\opt\java\jdk-25

SET PATH=%JAVA_HOME%\bin;%PATH%
SET PATH=%PATH%;C:\opt\llama.cpp-b6527-bin
SET GGML_BACKEND_PATH=C:\opt\llama.cpp-b6527-bin

cd /d C:\projects\llama-cpp-jna

call mvn clean install -DskipTests
call mvn dependency:copy-dependencies -DoutputDirectory=target/lib -pl examples

call java ^
  -cp "examples\target\examples-1.0.6527.0.jar;examples\target\lib\*" ^
  com.quasarbyte.llama.cpp.jna.examples.simplechat.SimpleChat ^
  -m C:\opt\models\Qwen3-8B-Q8_0.gguf -c 32768 -ngl 100
