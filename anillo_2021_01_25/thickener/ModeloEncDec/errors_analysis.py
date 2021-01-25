import torch

folders = ['bed/figures/', 'pressure/figures/', 'torque/figures/', 'solidC/figures/']
mean_values = [4.89, 96.86, 23.43, 69.2]

training_errors = [torch.load(folders[i] + 'errors.pkl')['training_errors'] for i in range(len(folders))]
eval_errors = [torch.load(folders[i] + 'errors.pkl')['eval_errors'] for i in range(len(folders))]

for i in range(len(folders)):
    print('{}: {} %'.format(folders[i], round(eval_errors[i][-1]/mean_values[i]*100, 2)))