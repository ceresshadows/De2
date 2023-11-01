from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device=0 if device.type == "cuda" else -1  # 0 for CUDA, -1 for CPU
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")


# from transformers import AutoTokenizer, LlamaForCausalLM

# # Initialize the tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\lenovo\\Desktop\\hf_llama")
# model = LlamaForCausalLM.from_pretrained("C:\\Users\\lenovo\\Desktop\\hf_llama")

# # Initial prompt
# prompt = 'Hey, what a nice day. Can you talk to me?'

# # Loop for 20 conversations
# for i in range(1, 21):
#     print(f'Round {i} - Input: {prompt}')
    
#     # Tokenize the prompt and generate a response
#     inputs = tokenizer(prompt, return_tensors='pt')
#     input_ids = inputs.input_ids
#     output_ids = model.generate(input_ids, max_length=200)
    
#     # Remove the input_ids from the output
#     response_ids = output_ids[0][input_ids.shape[-1]:]
    
#     # Decode the output and print it
#     output = tokenizer.decode(response_ids, skip_special_tokens=True)
#     print(f'Round {i} - LLaMA says: {output}\n')
    
#     # Use the output as the next prompt
#     prompt = output
