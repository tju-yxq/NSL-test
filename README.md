### **任务一：基础 GPT-2 模型实现**

#### **项目简介**

本项目旨在从零开始，通过填充核心函数来构建一个功能完整的 GPT-2 模型。这是一个基础的自回归语言模型，基于 Transformer 的“仅解码器”（Decoder-only）架构。通过完成此任务，您将深入理解构成现代大语言模型的关键技术组件，如自注意力机制、前馈网络和层归一化等。

#### **实现思路**

本次任务的核心是补全 `NSL-gpt2.py` 文件中的七个关键函数。您需要根据文档说明、函数注释以及相关的理论知识（如《Attention Is All You Need》论文），使用 `torch` API 来完成这些函数的内部逻辑：

  * **`gelu`**: 实现 GELU 激活函数，为模型引入非线性。
  * **`softmax`**: 将任意分数转换为概率分布，是注意力机制的核心。
  * **`layer_norm`**: 实现层归一化，稳定训练过程。
  * **`linear`**: 构建基础的全连接层，用于数据变换。
  * **`ffn`**: 结合 `linear` 和 `gelu`，构建前馈网络，用于信息加工。
  * **`attention`**: 实现缩放点积注意力，计算单个注意力头的输出。
  * **`mha`**: 管理多个 `attention` 头，实现多头自注意力机制，并应用因果掩码以防止“偷看未来”。

#### **运行方式**

完成所有函数填充后，您可以通过以下命令运行程序：

```bash
python NSL-gpt2.py "Alan Turing theorized that computers would one day become" --n_tokens_to_generate 50
```

#### **预期成果**

成功运行后，程序将使用 124M 模型进行自回归推理，并生成与任务文档要求完全一致的文本。这证明了您对 Transformer 解码器核心组件的实现是完全正确的。

  * **生成结果**:
    ```
     the most powerful machines on the planet.

    The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain. The
    ```
  * **性能观察**: 您会注意到，这是一个功能正确但速度较慢的基础实现，为后续任务二和任务三的性能优化奠定了基础。


