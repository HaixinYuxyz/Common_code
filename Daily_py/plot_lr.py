# Update the function for learning rate with a maximum learning rate of 0.0008
max_lr = 0.0008
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rcParams
import numpy as np

config = {
    "font.family": 'serif',
    "font.size": 18,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
# Define the parameters
milestones = [30, 50, 70, 90]
warmup_epochs = 20
epochs = range(100)


def learning_rate_schedule_with_max_lr(epoch):
    if epoch <= warmup_epochs:
        return (epoch / warmup_epochs) * max_lr
    else:
        return max_lr * (0.25 ** len([m for m in milestones if m <= epoch]))


# Update learning rates with the new function
learning_rates_with_max_lr = [learning_rate_schedule_with_max_lr(epoch) for epoch in epochs]

# Plotting with the updated learning rates

plt.figure(figsize=(10, 6))
plt.plot(epochs, learning_rates_with_max_lr, marker='o', color='green')
plt.title('学习率设置')
plt.xlabel('Epoch')
plt.ylabel('学习率')
plt.grid(True)

plt.show()
