import numpy as np
import math 
  
def sigmoid(x):
  return 1/(1 + np.exp(-x)) 



bias_output_layer = 0.10646543651819229125976562500000
bias_forget = np.genfromtxt('bias_forget_final', delimiter=' ')
bias_input = np.genfromtxt('bias_input_final', delimiter=' ')
bias_memory_cell = np.genfromtxt('bias_memory_cell_final', delimiter=' ')
bias_output = np.genfromtxt('bias_output_final', delimiter=' ')

weights_forget_gate = np.genfromtxt('weights_forget_gate_final', delimiter=' ')
weights_forget_hidden = np.genfromtxt('weights_forget_hidden_final', delimiter=' ')
weights_input_gate = np.genfromtxt('weights_input_gate_final', delimiter=' ')
weights_input_hidden = np.genfromtxt('weights_input_hidden_final', delimiter=' ')
weights_memory_cell = np.genfromtxt('weights_memory_cell_final', delimiter=' ')
weights_memory_cell_hidden = np.genfromtxt('weights_memory_cell_hidden_final', delimiter=' ')
weights_output = np.genfromtxt('weights_output_final', delimiter=' ')
weights_output_gate = np.genfromtxt('weights_output_gate_final', delimiter=' ')
weights_output_hidden = np.genfromtxt('weights_output_hidden_final', delimiter=' ')

window = [-0.59011993, -0.58965256, -0.59124217, -0.5905213, -0.59232481, -0.58745033, -0.58735262, -0.58673737]

input = window[0]
output = [0] * 256

input_gate_matrix = input*weights_input_gate + np.matmul(output, weights_input_hidden) + bias_input
input_gate = sigmoid(input_gate_matrix)

print(input_gate_matrix)