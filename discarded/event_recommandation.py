from transformers import AutoTokenizer, pipeline
import json

# Initialize the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Initialize the text-generation pipeline
text_gen = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_length=600)

# Define the initial prompt focusing on "events" in time series
initial_prompt = "[INST] <<SYS>> You are a virtual data scientist specialized in feature engineering for causal discovery in traffic time series data. Your task is to identify 'events' such as change points, interaction points between multiple series, or change points after interactions. \
The raw data is in JSON format as a time series. Please recommend a event defination in a JSON-like format, represented by a simple NumPy formula. \
For example, {'event_name': 'vehicle speed change point', 'formula': 'np.diff(position) > threshold'}. \
<</SYS>> Please recommend 3 events for identifying 'events' in time series data. [/INST]"

# Create a dictionary to store the recommended events
events_dict = {}

# Run the loop for 5 iterations
for i in range(3):
    # Generate a response
    response = text_gen(initial_prompt)[0]['generated_text']
    # print()
    # Extract the event name and formula
    start_index = response.find("{'event_name': '") + 17
    end_index = response.find("', 'formula': '")
    event_name = response[start_index:end_index]
    
    start_index = response.find("', 'formula': '") + 14
    end_index = response.find("'}")
    formula = response[start_index:end_index]
    
    # Store the event name and formula in the dictionary
    events_dict[event_name] = formula
    
    # Update the prompt for the next iteration
    initial_prompt = f"[INST] <<SYS>> Based on the event definations you have recommanded, \
    can you suggest 3 additional event that involves a basic mathematical operation on this existing event? \
    Please keep the description and formula for the new event within 80 characters. \
    <</SYS>> Can you recommend an additional event based on the events you have recommanded? [/INST]"
# Print the final dictionary of recommended events
print(events_dict)
