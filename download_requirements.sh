echo "Downloading model checkpoints.. (about baselines via MiniLLM github repository)"
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vZ3B0Mi50YXI/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode)
wget -O gpt2.tar $DLINK
tar -xvf gpt2.tar
mv gpt2 checkpoints/
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vb3B0LnRhcj9zdj0yMDIzLTAxLTAzJnN0PTIwMjQtMDQtMTBUMTMlM0ExMSUzQTQ0WiZzZT0yMDUwLTA0LTExVDEzJTNBMTElM0EwMFomc3I9YyZzcD1yJnNpZz00Y1hKSWpWUlpISUJXcUhqUGdEbiUyRjAxb2N6cERXWGlwbVBDVWszWjh2YlElM0Q=" | base64 --decode)
wget -O opt.tar $DLINK
tar -xvf opt.tar
mv opt checkpoints/
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vbGxhbWEudGFyP3N2PTIwMjMtMDEtMDMmc3Q9MjAyNC0wNC0xMFQxMyUzQTExJTNBNDRaJnNlPTIwNTAtMDQtMTFUMTMlM0ExMSUzQTAwWiZzcj1jJnNwPXImc2lnPTRjWEpJalZSWkhJQldxSGpQZ0RuJTJGMDFvY3pwRFdYaXBtUENVazNaOHZiUSUzRA==" | base64 --decode)
wget -O llama.tar $DLINK
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
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vZGF0YS50YXI/c3Y9MjAyMy0wMS0wMyZzdD0yMDI0LTA0LTEwVDEzJTNBMTElM0E0NFomc2U9MjA1MC0wNC0xMVQxMyUzQTExJTNBMDBaJnNyPWMmc3A9ciZzaWc9NGNYSklqVlJaSElCV3FIalBnRG4lMkYwMW9jenBEV1hpcG1QQ1VrM1o4dmJRJTNE" | base64 --decode)
wget -O data.tar $DLINK
tar -xvf data.tar
DLINK=$(echo -n "aHR0cHM6Ly9jb252ZXJzYXRpb25odWIuYmxvYi5jb3JlLndpbmRvd3MubmV0L2JlaXQtc2hhcmUtcHVibGljL01pbmlMTE0vcHJvY2Vzc2VkX2RhdGEudGFyP3N2PTIwMjMtMDEtMDMmc3Q9MjAyNC0wNC0xMFQxMyUzQTExJTNBNDRaJnNlPTIwNTAtMDQtMTFUMTMlM0ExMSUzQTAwWiZzcj1jJnNwPXImc2lnPTRjWEpJalZSWkhJQldxSGpQZ0RuJTJGMDFvY3pwRFdYaXBtUENVazNaOHZiUSUzRA==" | base64 --decode)
wget -O processed_data.tar $DLINK
tar -xvf processed_data.tar
