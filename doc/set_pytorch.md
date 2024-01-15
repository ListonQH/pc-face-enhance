# Libtorch 环境配置

Libtorch的配置方法与OpenCV配置方法相似。

## 一、下载

从[PyTorch](https://pytorch.org/)官网下载zip压缩包，解压后完成环境配置即可。

{INSTALL PYTORCH}处选择如下：
1. PyTorch Build: Stable
2. Your OS: Windows
3. Package: LibTorch
4. Language: C++/Java
5. Compute Platform: 选择对应CUDA版本即可
6. Run this Command: 出现下载链接

## 二、系统环境变量配置：Path

在环境变量Path中，增加：path/to/libtorhch/lib

## 三、VS2022 开发环境配置

需要C++标准ISO C++17或者以上：属性 -> 配置属性 -> 常规 -> C++ 语言标准 -> 选择17或者以上。

Release、x64配置文件：
1. VC++/包含目录: path\to\libtorch\include
2. VC++/包含目录: path\to\libtorch\include\torch\csrc\api\include
3. VC++/库目录: path\to\libtorch\include\lib
4. 链接器/常规/附加库目录: path\to\libtorch\include\lib
5. 链接器/输入: path\to\libtorch\lib\\*.lib
