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
state = [0] * 256


for x in range(1):
  input_gate_matrix = window[x]*weights_input_gate + np.matmul(output, weights_input_hidden) + bias_input
  input_gate = sigmoid(input_gate_matrix)

  

  forget_gate_matrix = window[x]*weights_forget_gate + np.matmul(output, weights_forget_hidden) + bias_forget
  forget_gate = sigmoid(forget_gate_matrix)

  output_gate_matrix = window[x]*weights_output_gate + np.matmul(output, weights_output_hidden) + bias_output
  output_gate = sigmoid(output_gate_matrix)

  memory_cell_matrix = window[x]*weights_memory_cell + np.matmul(output, weights_memory_cell_hidden) + bias_memory_cell
  memory_cell = np.tanh(memory_cell_matrix)

  state = state * forget_gate + input_gate * memory_cell
    
  output = output_gate * np.tanh(state)
 # output = output_gate * state

input_mult_test = window[0]*weights_input_gate

prediction = np.matmul(output, weights_output) + bias_output_layer
print("prediction")
print(prediction)
#print("input gate matrix")
#print(input_gate_matrix)
print("input_gate")
print(input_gate)
#print("output")
#print(output)
