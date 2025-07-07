## 1.taransformers库推理
使用python脚本：

```python
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------- Step 1: 接收命令行参数 ----------
parser = argparse.ArgumentParser(description="Run Qwen inference with a custom prompt and model")
parser.add_argument("--prompt", type=str, required=True, help="Prompt text for inference")
parser.add_argument("--model", type=str, required=True, help="Path to the model directory")
args = parser.parse_args()
prompt = args.prompt
model_name = args.model

# ---------- Step 2: 模型加载 ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("开始初始化模型")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("初始化模型完成")

# ---------- Step 3: 构造消息 ----------
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt},
]

# ---------- Step 4: 模型推理 ----------
print("开始推理")
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ---------- Step 5: 输出结果 ----------
print(f"结果: {response}")

```

启动时，使用

```python
python ***.py --model /mnt/Qwen/ --prompt "1+1等于几"
```

## 2.使用vllm启动在线推理服务
启动命令：

```python
CUDA_VISIBLE_DEVICES=0,1 vllm serve "/mnt/Qwen/Qwen/Qwen3-1.7B" --tensor-parallel-size 2 --enable-reasoning-parser --reasoning-parser deepseek_r1
```

半精度推理服务启动后，占用显存86G：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1748595638841-3b7ff991-791b-440d-a8b9-bc5cad03685b.png)

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1748596665980-37e2ad95-20ba-4500-9799-449738ce7999.png)

使用curl命令请求模型：

```python
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/mnt/Qwen/Qwen/Qwen3-1.7B",
  "messages": [
    {"role": "user", "content": "树上骑个猴，树下一个猴，问：总共几个猴？"}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "max_tokens": 8192,
  "presence_penalty": 1.5,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

在使用双卡推理如果报错nccl卡间通信错误，改成下面命令启动：

```python
NCCL_SOCKET_IFNAME=eth0 vllm serve "/mnt/Qwen/Qwen/Qwen3-1.7B" --tensor-parallel-size 2
```



**测试加载Deepseek-Qwen-Distill-32B模型**，全精度+gpu_memory_utilization (0.90)：

```python
NCCL_SOCKET_IFNAME=eth0 vllm serve "/mnt/zhangsh82/Qwen-Distill-32B" --tensor-parallel-size 2
```

载入模型权重过程耗费了较长时间，约12分钟

显存占用情况：

```python
model weights take 30.73GiB; non_torch_memory takes 0.74GiB;
PyTorch activation peak memory takes 1.41GiB;
the rest of the memory reserved for KV Cache is 53.19GiB.
```

推理速度：

```python
Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 47.5 tokens/s,
Running: 1 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.1%,
CPU KV cache usage: 0.0%.
```

## 3.基于SGlang框架推理Qwen3


使用modelscope下载模型：

```python
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local_dir /mnt/zhangsh82/Qwen
```

### 1.安装sglang环境
```python
pip install uv
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]==0.4.4.post1" --no-deps#pytorch2.6安装0.4.2不报错

pip install ipython
pip install orjson

```

### 2.启动模型命令
```python
python -m sglang.launch_server --model-path /mnt/Qwen/Qwen/Qwen3-1.7B
```

## 4.基于verl框架训练Qwen3(强化学习)
1.安装环境

```python
git clone https://github.com/volcengine/verl.git
cd verl

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple ##在镜像自带环境下补充安装需要的包

pip install deepspeed==0.16.9 -i https://pypi.tuna.tsinghua.edu.cn/simple ##解决deepspeed不匹配问题
#强化学习的
## 数据集下载
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=/mnt/zhangsh82/verl python3 examples/data_preprocess/gsm8k.py --local_dir /mnt/zhangsh82/verl/data/gsm8k
```





<font style="background-color:#DF2A3F;">※强化学习中由于依赖于模型输出，所以对模型输出的格式有着严重依赖，即对prompt有较高要求</font>

而本次中的prompt是写在数据集内的，在拉取数据集时，将prompt拼接到了问题的最后面，代码位置：

 /mnt/zhangsh82/verl/examples/data_preprocess/gsm8k.py    
![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751859993678-36064813-ed31-4670-ab32-5acf7553248d.png)

我这里将其改成了：

```python
instruction_following = '''/no_thinkPlease solve the problem step by step using concise reasonii
ng and minimal explanation.
Only include necessary math steps and avoid repeating the question.
Conclude your response with the final answer in the format:

#### <final_answer>

For example:

Question:
If there are 20 apples and you eat 5, how many are left?

Answer:
20 - 5 = 15
#### 15'''
```

<font style="background-color:#FBDE28;">PS:后面训练过程中，第一次模型正常按格式输出了答案，但是经过训练后，又开始格式混乱，这个严重影响模型的效果评估。</font>

查看训练集中的prompt：

```python
import pandas as pd
df = pd.read_parquet('/mnt/zhangsh82/verl/data/gsm8k/train.parquet')
print(df.columns)
print(df.iloc[0]['prompt'])

#测试集
import pandas as pd
df = pd.read_parquet('/mnt/zhangsh82/verl/data/gsm8k/test.parquet')
print(df.columns)
print(df.iloc[0]['prompt'])
```



2.启动训练

先尝试使用两张阿里PPU显卡（96GB * 2），跑Qwen3-1.7B + GRPO

```python
NCCL_SOCKET_IFNAME=eth0 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/zhangsh82/verl/data/gsm8k/train.parquet \
    data.val_files=/mnt/zhangsh82/verl/data/gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="/mnt/Qwen/Qwen/Qwen3-1.7B" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.actor.optim.warmup_steps=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_1.7b_function_rm' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=90 \
    trainer.test_freq=45 \
    trainer.total_epochs=1 \
    +actor_rollout_ref.ref.fsdp_config.optimizer_offload=True \
    +trainer.save_dir="/mnt/zhangsh82/trained_model/checkpoints/qwen3_1.7b" \
    +critic.ppo_micro_batch_size_per_gpu=4 \
    ++critic.model.path="/mnt/Qwen/Qwen/Qwen3-1.7B" \
    +reward_fn=rm \
    +reward_model.path="/mnt/zhangsh82/Qwen-Distill-32B" \
    +reward_model.tokenizer_path="/mnt/zhangsh82/Qwen-Distill-32B" \
    +trainer.dtype=half \
    +actor_rollout_ref.model.dtype=half \
    +actor_rollout_ref.ref.model.dtype=half \
    +actor_rollout_ref.rollout.model.dtype=half \
    +actor_rollout_ref.actor.fsdp_config.dtype=half \
    +actor_rollout_ref.ref.fsdp_config.dtype=half \
    ++actor_rollout_ref.rollout.dtype=half \
    +actor_rollout_ref.ref.dtype=half \
    +trainer.resume_from_checkpoint=True \
    +trainer.resume_path="/mnt/zhangsh82/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_1.7b_function_rm" 
```

中途提示内存（64GB）不够，ray的工作线程占用超过了95%，考虑缩小参数或改为小模型，或加内存

增加内存到128GB后重新开始训练

保存checkpoints：

```python
local_global_step_folder: checkpoints/verl_grpo_example_gsm8k/qwen3_1.7b_function_rm/global_step_90
(WorkerDict pid=1833) INFO:2025-07-01 10:54:59,613:[Rank 0] Saved model to /mnt/zhangsh82/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_1.7b_function_rm/global_step_90/actor/model_world_size_2_rank_0.pt
```

训练过程中<font style="color:#000000;background-color:#FBDE28;">显存占用（约使用156GB）</font>：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751338678294-44206422-b8a5-482c-ab0b-f1493371385c.png)

机器<font style="background-color:#FBDE28;">内存占用</font>：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751338838574-d558c32f-4ec5-4691-b881-985a1607b048.png)



verl框架基于fsdp训练后，不能直接加载使用模型。需要进行权重转换，verl提供了转换代码：

```python
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir "/mnt/zhangsh82/verl/checkpoints/verl_grpo_example_gsm8k/qwen3_1.7b_function_rm/global_step_1080/actor" \
  --target_dir "/mnt/zhangsh82/trained_model/Qwen3-1.7B-verl-no_reward_model"
```

成功：

![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751353928305-349b2c77-67b8-4999-97d1-584a45e24852.png)

测试RL后模型性能：

```python
git clone https://gitee.com/xzgan/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
```

```python
lm_eval --model hf \
    #verl+grpo训练后模型
    #--model_args pretrained=/mnt/zhangsh82/trained_model/Qwen3-1.7B-verl-no_reward_model \
    #原版qwen3-1.7B模型
    --model_args pretrained=/mnt/Qwen/Qwen/Qwen3-1.7B \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --num_fewshot 2 \
    --limit 100 \  #测试数据100条
    --output_path /mnt/zhangsh82/trained_model/eval/
```

```python
lm_eval --model hf \
    --model_args pretrained=/mnt/Qwen/Qwen/Qwen3-1.7B \
    --tasks gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --num_fewshot 2 \
    --limit 100 \
    --output_path /mnt/zhangsh82/trained_model/eval/
```

模型效果得分（gsm8k测试集）

**BaseLine：**  
原版Qwen3-1.7B模型测试得分（100条测试数据）：  
![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751537234618-da0e8412-5aa7-4178-bf59-512090dea858.png)

原版Qwen3-1.7B模型测试得分（全量测试数据）：  
![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751537270059-aad7c85b-c183-4826-a7ed-26779e0a3f03.png)



**训练后模型：**

无KL惩罚+无奖励模型训练的模型测试得分：  
![](https://cdn.nlark.com/yuque/0/2025/png/38450811/1751537198595-083d3f84-764d-4d47-b3b3-ac2777fbd353.png)

