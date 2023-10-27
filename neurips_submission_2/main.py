from helper import HFllama,Bloom,Qwen
import torch
from fastapi import FastAPI

import logging
import time

import torch

torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

lora_path='Percent-BFD/nips_qwen14b_lora_v7'
model_path = 'Qwen/Qwen-14B'
device='auto'

model = Qwen(
    model_path=model_path,
    tokenizer_path=model_path,
    device=device,
    if_bf16=True,
    lora_weights=lora_path,
)

if device == 'auto':
    device = 'cuda'

LLAMA2_CONTEXT_LENGTH = 4096


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    
    encoded = model.tokenizer(input_data.prompt, return_tensors="pt")
    
    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    t0 = time.perf_counter()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )
    
    t = time.perf_counter() - t0

    output = model.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
        idx, val = tlp
        tok_str = model.tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=model.tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()
    output = output[len(input_data.prompt):]
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = model.tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)

