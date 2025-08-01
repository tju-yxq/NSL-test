import numpy as np
import torch
import time
import math
torch.set_printoptions(8)

def gelu(x):
    """
        Task: Use the torch API to implement the approximate calculation formula of the `GELU`
        activation function. The formula is as follows (you need to paste it into the latex
        online conversion website)
        Website: https://www.latexlive.com/
        Formula: \frac{1}{2} x\left[1+\tanh \left(\sqrt{\frac{2}{\pi}}\left(x+0.044715 x^{3}\right)\right)\right]
        
        Input: Tensor
        Output: Tensor
    """
    # 直接按照提示写出返回函数值，这是一个对线性层输出进行非线性变换的激活函数
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    # softmax函数是将任意数值转换为概率分布
    # 1. 减去最大值
    x = x - torch.max(x, dim=-1, keepdim=True)[0]
    # 2. 计算每个元素的指数。
    ex = torch.exp(x)
    # 3. 计算分母，即所有指数化结果的和。
    sum_ex = torch.sum(ex, dim=-1, keepdim=True)
    # 4. 返回最终的概率分布。
    return ex / sum_ex
    


def layer_norm(x, g_b, eps:float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input: 
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """
    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])

    # layer_norm函数实现层归一化
    # 1. 计算均值和方差。
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    # 2. 归一化。根据 LayerNorm 公式 (x - mean) / sqrt(variance + eps)。
    #    eps 是为了防止方差为零导致除零错误。
    x_normalized = (x - mean) / torch.sqrt(variance + eps)
    # 3. 缩放和平移。将归一化的结果乘以可学习的缩放参数 g并加上偏移参数 b 。
    return x_normalized * g + b
    

def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer 
        Input: 
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = torch.Tensor(w_b['w']), torch.Tensor(w_b['b'])
    return x @ w + b
    
    

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input: 
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    # 按照 notes 实现前馈网络，总共三步过程
    # 1. 线性变换。使用 linear 函数
    a = linear(x, w_b1)
    # 2. GELU 激活。使用 gelu 函数
    a = gelu(a)
    # 3. 第二个线性变换
    x = linear(a, w_b2)

    return x
    


def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input: 
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    # 根据 papar编写 attention 函数。
    # 1. 计算 Q 和 K 的点积，得到原始的注意力分数。
    #    k.shape[-1] 即 d_k，是 key 向量的维度。
    d_k = k.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    # 2. 应用 mask。将 mask 中为 True 的位置设置为一个非常小的数 (-inf)。
    scores = torch.where(mask == 0, -torch.inf, scores)
    # 3. 应用 softmax，将分数转换为概率分布（注意力权重）。
    weights = softmax(scores)
    # 4. 将权重与 V 相乘，得到加权的输出。
    return weights @ v
    

def mha(x, attn, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention
        
        Input: 
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    # qkv projection，利用 linear 创建 Q, K, V
    x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]
    
    # Split into qkv
    """
        Task: Split the q,k,v matrix from the tensor x
        Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
    """

    # 调用 torch.split 将张量沿最后一个维度分割成3块。
    qkv = x.split(x.shape[-1] // 3, dim=-1)

    # Split into heads
    qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
    qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]

    # Causal mask to hide future inputs from being attended to
    """
        Task: Construct mask matrix
        Notes: 
            | 0  -inf -inf ... -inf |
            | 0    0  -inf ... -inf |
            | 0    0    0  ... -inf |
            |...  ...  ... ...  ... | 
            | 0    0    0  ...   0  |
        Mask is a tensor whose dimension is [n_seq, n_seq]
    """
    # 构建因果掩码矩阵。
    # 1. torch.ones(...) 创建一个全1的方阵。
    # 2. torch.tril(...) 取这个方阵的下三角部分，其余部分为0。这样就得到了一个只有下三角和对角线是1的掩码。
    n_seq = x.shape[0]
    causal_mask = torch.tril(torch.ones(n_seq, n_seq))

    # Perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    
    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    # 利用 torch.cat 将所有头的输出在最后一个维度上拼接起来。
    x = torch.cat(out_heads, dim=-1)

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    return x


def transformer_block(x, block, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']
    
    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]
    
    x = torch.Tensor(x)
    # forward  through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, params, n_head=n_head)  # model forward 
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target,
                                n_tokens_to_generate, K):
    """
        Task: Load 124M and 1558M models at the same time, use greedy sampling, and complete speculative decoding

        Inputs:
            inputs (list): The initial list of token IDs from the prompt.
            draft_params, target_params: Model weights for the draft and target models.
            hparams_draft, hparams_target: Hyperparameters for both models.
            n_tokens_to_generate (int): The number of new tokens to generate.
            K (int): The number of tokens the draft model speculates at each step (e.g., 4).

        Returns:
            list: A list of newly generated token IDs.

    """
    from tqdm import tqdm

    # 投机性解码的核心实现
    # 维护一个最终确认的 token 序列 `confirmed_ids`。
    confirmed_ids = list(inputs)

    pbar = tqdm(total=n_tokens_to_generate, desc="Speculative Generating")

    while len(confirmed_ids) - len(inputs) < n_tokens_to_generate:

        # 1. 草稿阶段
        # ---------------------
        # 让小模型从当前已确认的序列出发，连续生成 K 个草稿 token。
        draft_ids = []
        draft_inputs = list(confirmed_ids)  # 复制当前已确认的序列作为小模型的输入
        for _ in range(K):
            # 调用原始的 gpt2 函数进行自回归生成
            logits_draft = gpt2(draft_inputs, draft_params, hparams_draft["n_head"])
            next_id_draft = np.argmax(logits_draft[-1])
            draft_inputs.append(int(next_id_draft))
            draft_ids.append(int(next_id_draft))

        # 2. 验证阶段
        # -----------------------
        # 让大模型进行并行计算来打分
        verify_inputs = confirmed_ids + draft_ids
        logits_target = gpt2(verify_inputs, target_params, hparams_target["n_head"])

        # 我们只关心大模型对草稿部分的验证结果，所以我们只取最后 K 个 token 对应的 logits
        target_predictions = np.argmax(logits_target[len(confirmed_ids) - 1:-1], axis=-1)

        # 3. 接受/拒绝阶段 (Accept/Reject)
        # -----------------------------
        # 逐一比较草稿和专家的意见。
        all_accepted = True
        for i in range(K):
            draft_token = draft_ids[i]
            target_token = target_predictions[i]

            if draft_token == target_token:
                # 匹配成功，接受该 token
                confirmed_ids.append(draft_token)
                pbar.update(1)
            else:
                # 出现不匹配，接受专家（大模型）的意见，并立刻中止本轮核对
                confirmed_ids.append(int(target_token))
                pbar.update(1)
                all_accepted = False
                break

        # 处理所有草稿都被接受的特殊情况。
        # 根据论文，如果所有 K 个草稿都被接受了，我们可以直接采纳大模型预测的第 K+1 个词。
        if all_accepted:
            last_token_from_target = np.argmax(logits_target[-1])
            confirmed_ids.append(int(last_token_from_target))
            pbar.update(1)

        # 检查是否已生成足够数量的 token
        if len(confirmed_ids) - len(inputs) >= n_tokens_to_generate:
            break

    pbar.close()

    # 返回新生成的部分
    return confirmed_ids[len(inputs):]

def main(prompt: str, n_tokens_to_generate: int = 50, K: int = 4, models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # 1. 加载小模型
    print("Loading draft model ...")
    encoder, hparams_draft, params_draft = load_encoder_hparams_and_params("355M", models_dir)

    # 2. 加载大模型
    print("Loading target model ...")
    # 注意：encoder 和 hparams 对于不同大小的模型是不同的，但分词器(encoder)是共享的
    _, hparams_target, params_target = load_encoder_hparams_and_params("1558M", models_dir)

    # 3. 对输入进行编码
    input_ids = encoder.encode(prompt)

    # 确保序列长度不会超限
    assert len(input_ids) + n_tokens_to_generate < hparams_target["n_ctx"]

    # 4. 调用投机性解码函数
    start = time.time()
    output_ids = greedy_speculative_generate(
        input_ids,
        params_draft,
        params_target,
        hparams_draft,
        hparams_target,
        n_tokens_to_generate,
        K
    )
    end = time.time()
    print(f"\nTime taken for Speculative Sampling ({n_tokens_to_generate} tokens, K={K}): {end - start:.2f}s")

    # 5. 解码
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)