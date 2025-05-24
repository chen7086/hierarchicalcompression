Install Dependencies：

conda create --name llm_autodl_a100 python=3.12.3

conda activate hc

pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

pip install \
  transformers==4.51.3 \
  datasets==3.5.0 peft==0.15.2 evaluate==0.4.3 nltk==3.9.1 openai==1.75.0 tiktoken==0.9.0 \
  sentence-transformers==4.1.0 sentence-splitter==1.4 munch==4.0.0 colorprint3==1.3a1 \
  triton jieba fuzzywuzzy rouge

use this to check install which kind of flashatt：
   import torch
   torch.compiled_with_cxx11_abi()


Usage:

1. Demo Test

python hc_demo.py

2. LongBench Benchmark:

2.1. Compression
python3 hc_longbench.py \
  --final_tokens 3000 \
  --hc_ratio 2 \
  --save_path result_hc_test.json

2.2. Fetch Answers
python evaluation/r_concurrency_fetch_responses.py \
  --input_json result_hc_test.json \
  --inference_model deepseek-chat \
  --openai_key sk-5b654f8bd0f346b3ba1e79e4ac2e6a68 \
  --answers_json answer_hc_test.json \
  --sample_size 0.25 \
  --max_workers 50

2.3. Evaluate Metrics
python evaluation/r_evaluate_metrics.py \
  --input_json result_hc_test.json \
  --answers_json answer_hc_test.json \
  --output_metrics metrics_hc_test.json \
  --target llm_answers
