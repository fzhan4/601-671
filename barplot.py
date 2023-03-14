import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.figure(figsize =(12, 6))

# set height of bar
models = ['distilbert-base-uncased\nbatch size=64', 'bert-base-uncased\nbatch size=32', 'bert-large-uncased\nbatch size=8', 
        'bert-base-cased\nbatch size=32', 'bert-large-cased\nbatch size=8', 'roberta-base\nbatch size=32', 'roberta-large\nbatch size=4']
dev_acc = [0.7116, 0.6217, 0.6217, 0.6413, 0.6217, 0.6217, 0.6217]
test_acc = [0.6874, 0.6426, 0.6236, 0.6454, 0.6237, 0.6405, 0.6111]

# Set position of bar on X axis
br1 = range(len(models))
br2 = [x + barWidth for x in br1]

# Make the plot
plt.bar(br1, dev_acc, width = barWidth, label ='dev acc')
plt.bar(br2, test_acc, width = barWidth, label ='test acc')

# Adding Xticks
plt.xlabel('Model used')
plt.ylabel('Accuracy')
plt.xticks([r + barWidth for r in range(len(models))], models, rotation=10)
plt.title('Model Comparison: Training and Validation Accuracy')
plt.legend()
fig.savefig('models_barplot.png')