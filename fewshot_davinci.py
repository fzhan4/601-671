import random
import openai
from datasets import load_dataset

# Set up OpenAI API credentials
openai.api_key = "OPENAI_API_KEY"

# Loading BoolQ dataset from HuggingFace
dataset = load_dataset('boolq')

#  form a prompt by randomly selecting 8 demonstrations from BoolQ (balanced)
while (True):
    demo_indices = random.sample(range(len(dataset['train'])), 8)
    demos = [dataset['train'][i] for i in demo_indices]
    if sum([demo['answer'] for demo in demos]) == 4:
        break

# Creating the prompt by interleaving 'yes' and 'no' demonstrations
prompt = ''
for i, demo in enumerate(demos):
    # Note, question comes first since we are doing question answering
    # and we don't wnt it to be truncated if the passage is too long
    prompt += demo['question'] + " [SEP] " + demo['passage'] + " [SEP] " + str(demo['answer']) + "\n"


# Testing the generator on a small subset of BoolQ instances
test_indices = random.sample(range(len(dataset['validation'])), 30)
instances = [dataset['validation'][i] for i in test_indices]
correct = 0

for instance in instances:
    # Creating the prompt for the instance by replacing one of the demonstrations with the instance
    prompt_instance = prompt + instance['question'] + " [SEP] " + instance['passage'] + " [SEP] "
    
    # Generating the prediction using GPT-3
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_instance,
        temperature=0.7,
        max_tokens=4,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    prediction = response['choices'][0].get('text').strip().split('\n')[0]
    # print('prediction:', prediction)
    # print('correct answer:', instance['answer'])
    
    # Checking if the prediction matches the correct answer
    if str(instance['answer']).lower() in prediction.lower():
        correct += 1

# Printing the accuracy of the generator on the test set
print(f'Accuracy: {correct/len(instances)*100:.2f}%')