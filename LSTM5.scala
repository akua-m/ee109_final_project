import spatial.dsl._


@spatial object LSTM5 extends SpatialApp { 

    type T = FixPt[TRUE, _16, _16]
    
    //Function: Tiled, general matrix-multiplication
    def matrix_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

            val c_sram = SRAM[T](a_operand.rows, b_operand.cols).buffer
            init(c_sram)

            val tileM = 16
            val tileN = 16
            val tileK = 16


            Foreach(a_operand.cols by tileK){kk =>
                val numel_k = min(tileK.to[Int], a_operand.cols - kk)
                Foreach(a_operand.rows by tileM par 1){mm =>
                    val numel_m = min(tileM.to[Int], a_operand.rows - mm)
                    Foreach(b_operand.cols by tileN par 2){nn =>
                        val numel_n = min(tileN.to[Int], b_operand.cols - nn)
                        val tileC = SRAM[T](16, 16).buffer
                        store_result(c_sram(mm::mm+numel_m, nn::nn+numel_n), tileC)

                        // Your code here
                        MemFold(tileC)(numel_k by 1 par 1) { k =>
                            val temp = SRAM[T](tileM, tileN)
                            Foreach(numel_m by 1 par 1) { m =>
                              Foreach(numel_n by 1 par 1) { n =>
                                temp(m, n) = a_operand(mm + m, kk + k) * b_operand(kk + k, nn + n)
                              }
                            }

                            temp
                        }{_+_}        

                        store_result(tileC, c_sram(mm::mm+numel_m, nn::nn+numel_n))
                        
                    }
                }
            }
        }
    }

    //Function: initializes all elements of passed SRAM to 0 (zero)
    def init(input: SRAM2[T]) : Unit = {

        Foreach(input.rows.to[Int] by 1) { r =>
            Foreach(input.cols.to[Int] by 1) { c =>
                input(r, c) = 0.to[T]
            }
        }

    }

    //Function: element-wise multiplication of all elements in two SRAMs
    def element_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = a_operand(r, c) * b_operand(r, c)
            }
        }
        c_sram

    }


    //Function: element-wise addition of all elements in two SRAMs
    def element_add(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = a_operand(r, c) + b_operand(r, c)
            }
        }
        c_sram

    }


    //Function: calculates the sigmoid of passed value. Source: 
    def sigmoid (a_operand: T): T = {

        val a_sigmoid = Reg[T](0)

        if (a_operand < -2.5.to[T]){
            a_sigmoid := 0.to[T]
        } else if (a_operand > 2.5.to[T]) {
            a_sigmoid := 1.to[T]
        } else {
            a_sigmoid := (0.2.to[T] * a_operand + 0.5.to[T])
        }
        a_sigmoid.value  

    }


    //Function: element-wise sigmoid calculation of all elements in SRAM
    def element_sigmoid(a_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)

        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r,c) = sigmoid(a_operand(r,c))
            }
        }

        c_sram

    }

    //Function: element-wise tanh calculation of all elements in SRAM
    def element_tanh(a_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = 2 * sigmoid(2 * a_operand(r,c)) - 1
            }
        }
        
        c_sram
        
    }


    //Function used to copy the elements of 'resultant' into 'store_sram'
    def store_result (resultant: SRAM2[T], store_sram: SRAM2[T]) : Unit = {
        
        Foreach(resultant.rows.to[Int] by 1) { r =>
            Foreach(resultant.cols.to[Int] by 1) { c =>
                store_sram(r, c) = resultant(r, c)
            }
        }

    }


    //Defines the LSTM Cell. Takes the input gate calculations to calculate the state and output, which are propogated
    //to the next cell until the final prediction is made
    def LSTM_Cell (arg_input_gate: SRAM2[T], arg_forget_gate: SRAM2[T], arg_output_gate: SRAM2[T], arg_memory_cell: SRAM2[T], arg_state: SRAM2[T], arg_output: SRAM2[T]): Unit =  {
       
        val state = element_add(element_mult(arg_state, arg_forget_gate), element_mult(arg_input_gate, arg_memory_cell))
        store_result(state, arg_state)
       
        val output = element_mult(arg_output_gate, element_tanh(state))
        store_result(output, arg_output)
 
    } 


    
    def main (args: Array[String]): Unit = {

        //Register for prediciton
        val argRegOut = ArgOut[T]
    
    	//Read pre-calculated weights from CSV files, load onto DRAMs
        val bias_forget_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/bias_forget_final.csv", ",")
        val bias_f_mem = DRAM[T](1, 256)
        setMem(bias_f_mem, bias_forget_array)
            
        val bias_input_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/bias_input_final.csv", ",")
        val bias_in_mem = DRAM[T](1, 256)
        setMem(bias_in_mem, bias_input_array)
        
        val bias_memory_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/bias_memory_cell_final.csv", ",")
        val bias_m_mem = DRAM[T](1, 256)
        setMem(bias_m_mem, bias_memory_array)
        
        val bias_output_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/bias_output_final.csv", ",")
        val bias_o_mem = DRAM[T](1, 256)
        setMem(bias_o_mem, bias_output_array)

        val weights_fg_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_forget_gate_final.csv", ",")
        val weights_fg_mem = DRAM[T](1, 256)
        setMem(weights_fg_mem, weights_fg_array)
        
        val weights_fgh_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_forget_hidden_final.csv", ",")
        val weights_fgh_mem = DRAM[T](256, 256)
        setMem(weights_fgh_mem, weights_fgh_array)
        
        val weights_i_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_input_gate_final.csv", ",")
        val weights_i_mem = DRAM[T](1, 256)
        setMem(weights_i_mem, weights_i_array)
        
        val weights_ih_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_input_hidden_final.csv", ",")
        val weights_ih_mem = DRAM[T](256, 256)
        setMem(weights_ih_mem, weights_ih_array)
        
        val weights_m_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_memory_cell_final.csv", ",")
        val weights_m_mem = DRAM[T](1, 256)
        setMem(weights_m_mem, weights_m_array)
        
        val weights_mh_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_memory_cell_hidden_final.csv", ",")
        val weights_mh_mem = DRAM[T](256, 256)
        setMem(weights_mh_mem, weights_mh_array)
        
        val weights_o_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_output_final.csv", ",")
        val weights_o_mem = DRAM[T](256, 1)
        setMem(weights_o_mem, weights_o_array)
        
        val weights_og_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_output_gate_final.csv", ",")
        val weights_og_mem = DRAM[T](1, 256)
        setMem(weights_og_mem, weights_og_array)
        
        val weights_oh_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/weights_output_hidden_final.csv", ",")
        val weights_oh_mem = DRAM[T](256, 256)
        setMem(weights_oh_mem, weights_oh_array)

        val window_array = loadCSV1D[T]("/home/akuam/spatial/apps/src/window.csv", ",")
        val window_mem = DRAM[T](7)
        setMem(window_mem, window_array)


        //for debugging: 
        //val result_one = DRAM[T](1, 256)
        //val result_two = DRAM[T](1, 256)
        //val result_three = DRAM[T](1, 256)
        //val result_four = DRAM[T](1, 256)
        //val state_dram = DRAM[T](1, 256)
        //val output_dram = DRAM[T](1, 256)
        //val prediction_dram = DRAM[T](1, 1)
        //val bias_output_dram = DRAM[T](1,1)

        Accel { 

        	//set memories: bring all matrices/data from DRAM to SRAMS

            val bias_output_layer = SRAM[T](1, 1)
            bias_output_layer(0,0) = 0.10646543651819229125976562500000.to[T]

            val bias_forget = SRAM[T](1, 256)
            bias_forget load bias_f_mem

            val bias_input = SRAM[T](1, 256)
            bias_input load bias_in_mem

            val bias_memory_cell = SRAM[T](1, 256)
            bias_memory_cell load bias_m_mem

            val bias_output = SRAM[T](1, 256)
            bias_output load bias_o_mem

            val weights_forget_gate = SRAM[T](1, 256)
            weights_forget_gate load weights_fg_mem
            
            val weights_forget_hidden = SRAM[T](256, 256)
            weights_forget_hidden load weights_fgh_mem

            val weights_input_gate = SRAM[T](1, 256)
            weights_input_gate load weights_i_mem

            val weights_input_hidden = SRAM[T](256, 256)
            weights_input_hidden load weights_ih_mem

            val weights_memory_cell = SRAM[T](1, 256)
            weights_memory_cell load weights_m_mem

            val weights_memory_cell_hidden = SRAM[T](256, 256)
            weights_memory_cell_hidden load weights_mh_mem

            val weights_output = SRAM[T](256, 1)
            weights_output load weights_o_mem

            val weights_output_gate = SRAM[T](1, 256)
            weights_output_gate load weights_og_mem

            val weights_output_hidden = SRAM[T](256, 256)
            weights_output_hidden load weights_oh_mem

            val window = SRAM[T](7)
            window load window_mem
        

            //Loop to scroll through input window and make predictions

            val state = SRAM[T](1, 256)
            init(state)
            val output = SRAM[T](1, 256)
            init(output)

            val input = SRAM[T](1, 1)


            Sequential.Foreach(7 by 1) { a =>
                input(0, 0) = window(a)

                val input_gate_matrix = element_add(element_add(matrix_mult(input, weights_input_gate), matrix_mult(output, weights_input_hidden)), bias_input)        
                val input_gate1 = element_sigmoid(input_gate_matrix)
                //for debugging: result_two store input_gate
          
                val forget_gate_matrix= element_add(element_add(matrix_mult(input, weights_forget_gate), matrix_mult(output, weights_forget_hidden)), bias_forget)
                val forget_gate1 = element_sigmoid(forget_gate_matrix)
                //for debugging: result_three store forget_gate
                    
                val output_gate_matrix = element_add(element_add(matrix_mult(input, weights_output_gate), matrix_mult(output, weights_output_hidden)), bias_output)
                val output_gate1 = element_sigmoid(output_gate_matrix)
                //for debuggin: esult_four store output_gate

                val memory_cell_matrix = element_add(element_add(matrix_mult(input, weights_memory_cell), matrix_mult(output, weights_memory_cell_hidden)), bias_memory_cell)
                val memory_cell1 = element_tanh(memory_cell_matrix)

                LSTM_Cell(input_gate, forget_gate, output_gate, memory_cell, state, output)
                //for debugging: state_dram store state
                //for debugging: output_dram store output

            } //end Sequential

            val prediction = (matrix_mult(output, weights_output))(0,0) + bias_output_layer(0,0)
            argRegOut := prediction
            
        } //end Accel
        

        println("prediction: " + argRegOut.value)

        //for debugging: 
        //printMatrix(getMatrix(result_one), "result_one: ")
        //printMatrix(getMatrix(result_two), "result_two: ")
        //printMatrix(getMatrix(result_three), "result_three: ")
        //printMatrix(getMatrix(result_four), "result_four: ")
        //printMatrix(getMatrix(bias_output_dram), "bias: ")
        //printMatrix(getMatrix(state_dram), "state: ")
        //printMatrix(getMatrix(output_dram), "output: ")
        //printMatrix(getMatrix(prediction_dram), "prediction: ")

    } //end main
}


/* 
// These functions were not used for the final implementation. 

//Function: Tiled, general matrix-multiplication
def matrix_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, b_operand.cols).buffer
        init(c_sram)

        val tileM = 16
        val tileN = 16
        val tileK = 16


        Foreach(a_operand.cols by tileK){kk =>
            val numel_k = min(tileK.to[Int], a_operand.cols - kk)
            Foreach(a_operand.rows by tileM par 2){mm =>
                val numel_m = min(tileM.to[Int], a_operand.rows - mm)
                Foreach(b_operand.cols by tileN par 2){nn =>
                    val numel_n = min(tileN.to[Int], b_operand.cols - nn)
                    val tileC = SRAM[T](16, 16).buffer
                    store_result(c_sram(mm::mm+numel_m, nn::nn+numel_n), tileC)

                    // Your code here
                    MemFold(tileC)(numel_k by 1 par 2) { k =>
                        val temp = SRAM[T](tileM, tileN)
                        Foreach(numel_m by 1 par 2) { m =>
                          Foreach(numel_n by 1 par 2) { n =>
                            temp(m, n) = a_operand(mm + m, kk + k) * b_operand(kk + k, nn + n)
                          }
                        }

                        temp
                    }{_+_}        

                    store_result(tileC, c_sram(mm::mm+numel_m, nn::nn+numel_n))
                    
                }
            }
        }
    }
}


// Function: calculate the tanh of passed value
// source: An Optimized Lookup-Table for the Evaluation of Sigmoid Function for Artificial Neural Networks
def tanh(a_operand: T): T = {
    val sigmoid_lut = LUT[T](15,3)(0.390625.to[T], 0.453125.to[T], 0.4049.to[T], 
                                    0.453125.to[T], 0.515625.to[T], 0.4558.to[T], 
                                    0.515625.to[T], 0.578125.to[T], 0.5038.to[T], 
                                    0.578125.to[T], 0.640625.to[T], 0.5490.to[T], 
                                    0.640625.to[T], 0.703125.to[T], 0.5911.to[T], 
                                    0.703125.to[T], 0.78125.to[T], 0.6348.to[T], 
                                    0.78125.to[T], 0.859375.to[T], 0.6791.to[T], 
                                    0.859375.to[T], 0.9375.to[T], 0.7190.to[T], 
                                    0.9375.to[T], 1.046875.to[T], 0.7609.to[T], 
                                    1.046875.to[T], 1.171875.to[T], 0.8057.to[T], 
                                    1.171875.to[T], 1.328125.to[T], 0.8493.to[T], 
                                    1.328125.to[T], 1.53125.to[T], 0.8916.to[T], 
                                    1.53125.to[T], 1.859375.to[T], 0.9329.to[T], 
                                    1.859375.to[T], 2.90625.to[T], 0.9740.to[T], 
                                    2.90625.to[T], 2.90625.to[T], 1.to[T])

    val a_tanh = Reg[T](0)

    val temp = Reg[T](0)
    val temp2 = Reg[T](0)

    Sequential { 

        if (a_operand < 0) {
            temp := -1 * a_operand
        } else {
            temp := a_operand
        }
            
        
        // Makes the assumption/approximation: tanh(x) (for x -> -inf) = x
        if (temp <= sigmoid_lut(0,0)){ 
            temp2 := temp
        } else if (sigmoid_lut(0,0) < temp && temp <= sigmoid_lut(0,1)) {
            temp2 := sigmoid_lut(0,2) 
        } else if (sigmoid_lut(1,0) < temp && temp <= sigmoid_lut(1,1)) {
            temp2 := sigmoid_lut(1,2) 
        } else if (sigmoid_lut(2,0) < temp && temp <= sigmoid_lut(2,1)) {
            temp2 := sigmoid_lut(2,2)
        } else if (sigmoid_lut(3,0) < temp && temp <= sigmoid_lut(3,1)) {
            temp2 := sigmoid_lut(3,2)
        } else if (sigmoid_lut(4,0) < temp && temp <= sigmoid_lut(4,1)) {
            temp2 := sigmoid_lut(4,2)
        } else if (sigmoid_lut(5,0) < temp && temp <= sigmoid_lut(5,1)) {
            temp2 := sigmoid_lut(5,2)
        } else if (sigmoid_lut(6,0) < temp && temp <= sigmoid_lut(6,1)) {
            temp2 := sigmoid_lut(6,2)
        } else if (sigmoid_lut(7,0) < temp && temp <= sigmoid_lut(7,1)) {
            temp2 := sigmoid_lut(7,2)
        } else if (sigmoid_lut(8,0) < temp && temp <= sigmoid_lut(8,1)) {
            temp2 := sigmoid_lut(8,2)
        } else if (sigmoid_lut(9,0) < temp && temp <= sigmoid_lut(9,1)) {
            temp2 := sigmoid_lut(9,2)
        } else if (sigmoid_lut(10,0) < temp && temp <= sigmoid_lut(10,1)) {
            temp2 := sigmoid_lut(10,2)
        } else if (sigmoid_lut(11,0) < temp && temp<= sigmoid_lut(11,1)) {
            temp2 := sigmoid_lut(11,2)
        } else if (sigmoid_lut(12,0) < temp && temp<= sigmoid_lut(12,1)) {
            temp2 := sigmoid_lut(12,2)
        } else if (sigmoid_lut(13,0) < temp && temp <= sigmoid_lut(13,1)) {
            temp2 := sigmoid_lut(13,2)
        } else {
            temp2 := sigmoid_lut(14,2)
        }
        

        if (a_operand < 0) {
            a_tanh := -1 * temp2.value
        } else {
            a_tanh := temp2.value
        }
    
    } //end Sequential

    a_tanh.value  

}
*/
