## 任务二：实现 KV Cache 加速推理

### 项目简介

本项目在任务一实现的基础GPT-2模型上，成功集成了**KV Cache**机制，旨在显著优化模型的自回归推理过程。通过缓存并重用先前计算过的键（Key）和值（Value）向量，本实现有效避免了大量的重复计算，从而在不牺牲任何生成质量的前提下，大幅降低了生成文本时的端到端时延。

### 实现思路

本次优化的核心思想是改变模型的数据流模式，从“每次都处理完整序列”变为“只处理新生成的词元，并从缓存中读取历史信息”。

为了实现这一点，我们对 `NSL-gpt2.py` 文件进行了系统性的修改，主要涉及以下几个核心函数：

  * **`generate`**: 作为总指挥，现在负责初始化并管理 `kv_cache`。它将处理`prompt`和生成新词的流程分开，在生成循环中，只将最新的 `token_id` 和完整的 `kv_cache` 传递给模型。
  * **`gpt2`**: 学会了接收 `kv_cache`，并根据缓存是否存在来智能地计算位置编码，只为新词元计算其在序列中的真实位置。
  * **`mha` (多头注意力)**: 能够区分处理两种情况下的掩码（Mask）：在处理初始`prompt`时使用因果掩码，在生成新词元时使用允许关注所有历史的掩码。
  * **`attention`**: 这是KV Cache机制最核心的执行点。它负责从 `kv_cache` 中读取历史的K,V值，将当前计算出的新K,V值与之拼接，并将更新后的、更长的K,V值存回缓存中。

### 运行方式

您可以使用与任务一完全相同的命令来运行此优化后的版本。

```bash
# 激活您的虚拟环境
# .\venv\Scripts\activate

# 运行程序
python NSL-gpt2.py "Alan Turing theorized that computers would one day become" --n_tokens_to_generate 50
```

### 预期成果

成功运行后，您将观察到以下两个关键成果：

1.  **结果一致性**: 生成的文本内容将与任务一中**完全相同**，证明了我们的优化是**无损**的。

    ```
     the most powerful machines on the planet.

    The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain. The
    ```

2.  **性能显著提升**: 您会注意到，终端中 `tqdm` 进度条显示的生成时间相比任务一有了**大幅度的降低**。这直观地证明了 KV Cache 策略在加速大语言模型推理方面的有效性。

    ```
    # 任务一的耗时 (示例)
    generating: 100%|██████████| 50/50 [00:13<00:00,  3.81it/s]
    Time taken to generate 50 tokens: 13.14s

    # 任务二使用 KV Cache 后的耗时 (示例)
    generating with KV Cache: 100%|██████████| 49/49 [00:02<00:00, 21.3it/s]
    Time taken to generate 50 tokens: 2.58s
    ```

通过本次任务，我们成功地将一个基础的GPT-2实现，升级为了一个带有高效推理缓存的优化版本。
