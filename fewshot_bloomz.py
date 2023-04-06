import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from huggingface_hub.inference_api import InferenceApi

API_TOKEN = "Huggingface-API-KEY"

# Set up bloomz
checkpoint = "bigscience/bloomz"

inference = InferenceApi(repo_id=checkpoint, token=API_TOKEN)

# Loading BoolQ dataset from HuggingFace
dataset = load_dataset('boolq')

#  form a prompt by randomly selecting 8 demonstrations from BoolQ (balanced)
while (True):
    demo_indices = random.sample(range(len(dataset['train'])), 4) # 8 is too long that exceeds input length 1000 tokens
    demos = [dataset['train'][i] for i in demo_indices]
    if sum([demo['answer'] for demo in demos]) == 2:
        break

# Creating the prompt by interleaving 'yes' and 'no' demonstrations
prompt = ''
for i, demo in enumerate(demos):
    # Note, question comes first since we are doing question answering
    # and we don't wnt it to be truncated if the passage is too long
    prompt += demo['question'] + " [SEP] " + demo['passage'] + " [SEP] " + str(demo['answer']) + "\n"


# Testing the generator on a small subset of BoolQ instances
N = 100
test_indices = random.sample(range(len(dataset['validation'])), N)
instances = [dataset['validation'][i] for i in test_indices]
correct = 0
dict_yn = {'yes': 'true', 'no': 'false'}

for instance in instances:
    # Creating the prompt for the instance by replacing one of the demonstrations with the instance
    prompt_instance = prompt + instance['question'] + " [SEP] " + instance['passage'] + " [SEP] "
    
    # Generating the prediction using bloomz
    response = inference(prompt_instance)
    # inputs = tokenizer.encode(prompt_instance, return_tensors="pt")
    # outputs = model.generate(inputs)
    # response = tokenizer.decode(outputs[0])
    
    if 'error' in response:
        N -= 1 # pass when input length exceeds 1000 tokens
    else: 
        prediction = response[0]['generated_text'].split()[-1]
        print('prediction:', prediction)
        print('correct answer:', instance['answer'])
        
        # Checking if the prediction matches the correct answer
        ans = str(instance['answer']).lower()
        if ans in prediction.lower() or (ans in dict_yn and dict_yn[ans] in prediction.lower()):
            correct += 1

# Printing the accuracy of the generator on the test set
print(f'Accuracy: {correct/N*100:.2f}%')