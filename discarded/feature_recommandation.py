# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# # Initialize the model and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Define the prompt
# prompt = "[INST] <<SYS>> You are a virtual data scientist specialized in feature engineering for causal discovery. Your task is to recommend features for analysis in the financial domain, specifically for stock market data. Please recommend up to 5 features in a JSON-like format, each represented by a simple NumPy formula. For example, {'feature_name': 'Stock highest value in first hour', 'formula': 'np.max(data[0:60])'}. <</SYS>> Please recommend features for stock market analysis in JSON-like format. [/INST]"

# # Generate a response
# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(inputs.input_ids, max_length=800, num_return_sequences=1)
# response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

# # Print the response
# print(response)


from transformers import AutoTokenizer, pipeline
import json

# Initialize the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the text-generation pipeline
text_gen = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_length=600)

# Define the initial prompt
initial_prompt = "[INST] <<SYS>> You are a virtual data scientist specialized in the financial domain. Your task is to recommend features for data analysis. \
    The raw data is in JSON format as a time series. \
    Please recommend a feature in a JSON-like format, by a NumPy formula (do not do compound calculations).\
    For example, {'feature_name': 'Stock highest value in first hour', 'formula': 'np.max(data[0:60])'}. \
    Please keep the length of each feature within 80. \
    <</SYS>> Please recommend 3 features for stock market analysis in JSON-like format. [/INST]"

# Create a dictionary to store the recommended features
features_dict = {}

# Run the loop for 5 iterations
for i in range(3):
    # Generate a response
    response = text_gen(initial_prompt)[0]['generated_text']
    print(f"Iteration {i+1} Response: {response}")

    # Extract the feature name and formula (this assumes a specific format in the response)
    # Note: This is a simple extraction method and may need adjustments based on actual model outputs.
    start_index = response.find("{'feature_name': '") + 17
    end_index = response.find("', 'formula': '")
    feature_name = response[start_index:end_index]
    
    start_index = response.find("', 'formula': '") + 14
    end_index = response.find("'}")
    formula = response[start_index:end_index]
    
    # Store the feature name and formula in the dictionary
    features_dict[feature_name] = formula
    
    # Update the prompt for the next iteration
    initial_prompt = f"[INST] <<SYS>> Based on the features you have recommanded, \
    can you suggest an additional feature that involves mathematical operation on this existing feature? \
    For example, {'feature_name': 'Stock highest value in first hour', 'formula': 'np.max(data[0:60])'}. \   
    Please keep the length of each feature within 80. \
    <</SYS>> Can you recommend an additional feature based on the features you have recommanded? [/INST]"
    
print("Final Recommended Features: ", features_dict)
# Print the final dictionary of recommended features
print(features_dict)



# from transformers import AutoTokenizer, pipeline
# import json

# # Initialize the model and tokenizer
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Initialize the text-generation pipeline
# text_gen = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_length=500)

# # Define the initial prompt
# prompt = "[INST] <<SYS>> You are a virtual data scientist specialized in feature engineering for causal discovery. \
# Your task is to recommend features for analysis in the financial domain, specifically for stock market data. \
# The raw data is in JSON format as a time series. Please recommend up to 5 features in a JSON-like format, \
# each represented by a simple NumPy formula.  \
# For example, {'feature_name': 'Stock highest value in first hour', 'formula': 'np.max(data[0:60])'}. \
# <</SYS>> Please recommend features for stock market analysis in JSON-like format. [/INST]"

# # Generate a response
# response = text_gen(prompt)[0]['generated_text']

# # Print the initial response
# print("Initial Response:")
# print(response)

# # Define the follow-up prompt for multi-turn conversation
# follow_up_prompt = "[INST] <<SYS>> Based on the features you've recommended, \
# can you suggest additional features that involve further mathematical operations on the existing ones? \
#     Please keep the description and formula for each feature within 80 characters. \
#         <</SYS>> Can you recommend additional features based on the existing ones? [/INST]"

# # Generate a follow-up response
# follow_up_response = text_gen(follow_up_prompt)[0]['generated_text']

# # Print the follow-up response
# print("Follow-up Response:")
# print(follow_up_response)
