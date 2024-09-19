echo "Downloading model checkpoints.. (about baselines via MiniLLM github repository)"
wget -O gpt2.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/gpt2.tar
tar -xvf gpt2.tar
mv gpt2 checkpoints/
wget -O opt.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/opt.tar
tar -xvf opt.tar
mv opt checkpoints/
wget -O llama.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/llama.tar
tar -xvf llama.tar
mv llama checkpoints/

echo "Chainging model path for interpretability.."
mv checkpoints/gpt2/train/kd/gpt2-base-xlarge-sft checkpoints/gpt2/train/kd/gpt2-base
mv checkpoints/gpt2/train/kd/gpt2-medium-xlarge-sft checkpoints/gpt2/train/kd/gpt2-medium
mv checkpoints/gpt2/train/kd/gpt2-large-xlarge-sft checkpoints/gpt2/train/kd/gpt2-large
mv checkpoints/gpt2/train/minillm/base-init-xlarge-sft checkpoints/gpt2/train/minillm/gpt2-base
mv checkpoints/gpt2/train/minillm/medium-init-xlarge-sft checkpoints/gpt2/train/minillm/gpt2-medium
mv checkpoints/gpt2/train/minillm/large-init-xlarge-sft checkpoints/gpt2/train/minillm/gpt2-large
mv checkpoints/gpt2/train/seqkd/gpt2-base-xlarge-sft checkpoints/gpt2/train/seqkd/gpt2-base
mv checkpoints/gpt2/train/seqkd/gpt2-medium-xlarge-sft checkpoints/gpt2/train/seqkd/gpt2-medium
mv checkpoints/gpt2/train/seqkd/gpt2-large-xlarge-sft checkpoints/gpt2/train/seqkd/gpt2-large
mv checkpoints/opt/train/kd/opt-1.3B-13B-sft checkpoints/opt/train/kd/opt-1.3B
mv checkpoints/opt/train/kd/opt-2.7B-13B-sft checkpoints/opt/train/kd/opt-2.7B
mv checkpoints/opt/train/kd/opt-6.7B-13B-sft checkpoints/opt/train/kd/opt-6.7B
mv checkpoints/opt/train/minillm/1.3B-init-13B-sft checkpoints/opt/train/minillm/opt-1.3B
mv checkpoints/opt/train/minillm/2.7B-init-13B-sft checkpoints/opt/train/minillm/opt-2.7B
mv checkpoints/opt/train/minillm/6.7B-init-13B-sft checkpoints/opt/train/minillm/opt-6.7B
mv checkpoints/opt/train/seqkd/opt-1.3B-13B-sft checkpoints/opt/train/seqkd/opt-1.3B
mv checkpoints/opt/train/seqkd/opt-2.7B-13B-sft checkpoints/opt/train/seqkd/opt-2.7B
mv checkpoints/opt/train/seqkd/opt-6.7B-13B-sft checkpoints/opt/train/seqkd/opt-6.7B
mv checkpoints/llama/train/kd/llama-7B-13B-sft checkpoints/llama/train/kd/llama-7B
mv checkpoints/llama/train/minillm/7B-init-13B-sft checkpoints/llama/train/minillm/llama-7B
mv checkpoints/llama/train/seqkd/llama-7B-13B-sft checkpoints/llama/train/seqkd/llama-7B

echo "Downloading model checkpoints.. (about PromptKD via huggingface hub)"
mkdir checkpoints/gpt2/train/promptkd
mkdir checkpoints/opt/train/promptkd
mkdir checkpoints/llama/train/promptkd
huggingface-cli download gmkim/GPT-2-120M-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/gpt2/train/promptkd/gpt2-base
huggingface-cli download gmkim/GPT-2-340M-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/gpt2/train/promptkd/gpt2-medium
huggingface-cli download gmkim/GPT-2-760M-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/gpt2/train/promptkd/gpt2-large
huggingface-cli download gmkim/OPT-1.3B-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/opt/train/promptkd/opt-1.3B
huggingface-cli download gmkim/OPT-2.7B-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/opt/train/promptkd/opt-2.7B
huggingface-cli download gmkim/OPT-6.7B-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/opt/train/promptkd/opt-6.7B
huggingface-cli download gmkim/Llama-7B-distilled-PromptKD-Dolly-v1.0 --local-dir checkpoints/llama/train/promptkd/llama-7B

echo "Downloading data for training and evaluation.."
wget -O data.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/data.tar
tar -xvf data.tar
wget -O processed_data.tar https://unilm.blob.core.windows.net/minillm/MiniLLM/processed_data.tar
tar -xvf processed_data.tar
