
### “QLoRA Optimization for Financial Risk Summarization and 
###         Classification on Earnings Calls and SEC Filings”

QLoRA: Efficient Finetuning of Quantized LLMs
https://arxiv.org/abs/2305.14314

https://openreview.net/pdf?id=aJnKjvTtPq
QLoRA: Efficient Finetuning of Quantized LLMs



ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
https://arxiv.org/abs/1910.02054

LoRa:
Low-Rank Adaptation of Large Language Models
https://arxiv.org/abs/2106.09685


Textbooks Are All You Need
https://arxiv.org/abs/2306.11644

LOW RANK QUANTIZATION ADAPTATION
https://openreview.net/pdf?id=aJnKjvTtPq


issues: 
Known Issues and Limitations

Here a list of known issues and bugs. If your issue is not reported here, please open a new issue and describe the problem.

    4-bit inference is slow. Currently, our 4-bit inference implementation is not yet integrated with the 4-bit matrix multiplication
    Resuming a LoRA training run with the Trainer currently not supported by HF.
    Currently, using bnb_4bit_compute_type='fp16' can lead to instabilities. For 7B LLaMA, only 80% of finetuning runs complete without error. We have solutions, but they are not integrated yet into bitsandbytes.
    Make sure that tokenizer.bos_token_id = 1 to avoid generation issues.
    If you get an this issue ("illegal memory access") then you should use a newer HF LLaMA conversion or downgrade your PyTorch version.
    Problems with adding new tokens outlined in #214. Embeddings need to be updated and stored/reloaded if you are adding new tokens.
