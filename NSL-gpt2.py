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
    


def attention(q, k, v, mask, kv_cache=None):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
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
    # 实现带有 KV Cache 的注意力机制。
    if kv_cache:
        # 1. 从 cache 中获取历史的 k 和 v。
        k_cache, v_cache = kv_cache['k'], kv_cache['v']
        # 2. 将当前 token 计算出的 k, v 与历史 cache 拼接。
        k = torch.cat([k_cache, k], dim=0)
        v = torch.cat([v_cache, v], dim=0)
        # 3. 将更新后的 k, v 存回 cache，供下一层或下一个时间步使用。
        kv_cache['k'] = k
        kv_cache['v'] = v

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


def mha(x, attn, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention

        Input:
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
            kv_cache: (Optional) A list of caches, one for each attention head.
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
    # 构建mask矩阵时区分处理
    n_seq_q = x.shape[0]  # 当前 query 的长度
    n_seq_past = kv_cache[0]['k'].shape[0] if kv_cache is not None else 0
    n_seq_k = n_seq_past + n_seq_q  # key/value 的总长度

    # 1. 先创建一个基础的二进制掩码 (1s and 0s)
    if n_seq_q > 1:
        # 情况一: 处理整个 prompt 时，需要一个下三角的因果掩码
        binary_mask = torch.tril(torch.ones(n_seq_q, n_seq_k))
    else:
        # 情况二: 生成单个 token 时，它可以关注所有历史 token
        binary_mask = torch.ones(1, n_seq_k)

    # 2. 将二进制掩码转换为加性掩码 (0.0 and -inf)
    #    这个掩码将直接加到 attention scores 上
    additive_mask = (1.0 - binary_mask) * -torch.inf

    # 3. 对每个头并行计算注意力，并传入修正后的掩码
    out_heads = [attention(q, k, v, additive_mask, kv_cache=kv_cache[i] if kv_cache is not None else None) for
                 i, (q, k, v) in enumerate(qkv_heads)]

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


def transformer_block(x, block, n_head, kv_cache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, kv_cache=kv_cache)

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

def gpt2(inputs, params, n_head, kv_cache=None):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']

    # 统一处理位置编码。
    n_seq_past = kv_cache[0][0]['k'].shape[0] if kv_cache is not None else 0
    positions = range(n_seq_past, n_seq_past + len(inputs))

    # token + positional embeddings
    x = wte[inputs] + wpe[positions]  # [n_seq] -> [n_seq, n_embd]

    x = torch.Tensor(x)

    # kv_cache 的结构是 [n_layer, n_head, {'k': tensor, 'v': tensor}]
    for i, block in enumerate(blocks):
        block_cache = kv_cache[i] if kv_cache else None
        x = transformer_block(x, block, n_head=n_head, kv_cache=block_cache)

    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, hparams, n_tokens_to_generate):
    from tqdm import tqdm

    n_layer = hparams['n_layer']
    n_head = hparams['n_head']
    n_embd = hparams['n_embd']

    # 1. 初始化一个空的 KV Cache 结构
    kv_cache = [
        [{'k': torch.empty(0, n_embd // n_head), 'v': torch.empty(0, n_embd // n_head)} for _ in range(n_head)]
        for _ in range(n_layer)
    ]

    # 我们需要一个变量来存储下一个要被处理的 token id。
    next_id = inputs[0]
    prompt_ids = inputs[1:]  # 将 inputs 分为第一个 token 和剩余的 prompt

    # 2. 首先，逐个处理输入的 prompt，填充初始的 KV Cache。
    for token_id in prompt_ids:
        # 输入永远是单个 token id
        gpt2([next_id], params, n_head, kv_cache)
        next_id = token_id

    # 3. 然后，开始真正的自回归生成循环。
    generated_ids = []
    for _ in tqdm(range(n_tokens_to_generate), "generating with KV Cache"):
        logits = gpt2([next_id], params, n_head, kv_cache)
        next_id = np.argmax(logits[-1])
        generated_ids.append(int(next_id))

    return generated_ids

def greedy_speculative_generate(inputs, draft_params, target_params, hparams_draft, hparams_target, n_tokens_to_generate, K):
    
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
    generated_ids = []
    current_inputs = list(inputs)

    while len(generated_ids) < n_tokens_to_generate:
        return generated_ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not suring the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams, n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire
    fire.Fire(main)