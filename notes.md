block arch:

mamba
mamba moe
mamba
mamba moe
transformer
mamba moe
mamba 
mamba moe


layer arches

mamba:
rmsnorm
mamba
add
rmsnorm
mlp
add

mambamoe:
rmsnorm
mamba
add
rmsnorm
moe
add

transformers:
rmsnorm
attention
add
rmsnorm
mlp
add

transformersmoe:

rmsnorm
attention
add
rmsnorm
moe
add

Other architecture details are standard, including grouped-query attention (GQA), SwiGLU activation
function [ 7, 45 , 50 ], and load balancing for the MoE [14 ]. The vocabulary size is 64K. The tokenizer
is trained with BPE [16 , 33 , 44 ] and each digit is a separate token [7 ]. We also remove the dummy
space used in Llama and Mistral tokenizers for more consistent and reversible tokenization



vocab size is 64k

read more on gqa and replace mha with it

add proper moe stuff

tokenizer work
