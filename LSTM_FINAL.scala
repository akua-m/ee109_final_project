import spatial.dsl._


@spatial object LSTM extends SpatialApp { 

    type T = FixPt[TRUE, _16, _16]

    
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

        c_sram

    } //end matrix_mult

    def init (input: SRAM2[T]) : Unit = {
        
        Foreach(input.rows.to[Int] by 1) { r =>
            Foreach(input.cols.to[Int] by 1) { c =>
                input(r, c) = 0.to[T]
            }
        }
    } //end init


    def element_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = {  

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = a_operand(r, c) * b_operand(r, c)
            }
        }

        c_sram

    } //end element_mult


    def element_add(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = a_operand(r, c) + b_operand(r, c)
            }
        }

        c_sram

    } //end element_add

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

    } //end sigmoid

    def element_sigmoid(a_operand: SRAM2[T]) : SRAM2[T] = { 

        //sigmoid formula: sigmoid(x) = (tanh(x/2) + 1) / 2

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)

        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r,c) = sigmoid(a_operand(r,c))
            }
        }

        c_sram

    } //end element_sigmoid


    def element_tanh(a_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, a_operand.cols)
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = 2 * sigmoid(2 * a_operand(r,c)) - 1
            }
        }
        
        c_sram
        
    } //end element_tanh


    def store_result (resultant: SRAM2[T], store_matrix: SRAM2[T]) : Unit = {
        
        Foreach(resultant.rows.to[Int] by 1) { r =>
            Foreach(resultant.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = resultant(r, c)
            }
        }
    } //end store_result


    def LSTM_Cell (arg_input_gate: SRAM2[T], arg_forget_gate: SRAM2[T], arg_output_gate: SRAM2[T], arg_memory_cell: SRAM2[T], arg_state: SRAM2[T], arg_output: SRAM2[T]): Unit =  {
        //state = state * forget_gate + input_gate * memory_cell   
        //output = output_gate * tf.tanh(state)
       
        val state = element_add(element_mult(arg_state, arg_forget_gate), element_mult(arg_input_gate, arg_memory_cell))
        store_result(state, arg_state)
       
        val output = element_mult(arg_output_gate, element_tanh(state))
        store_result(output, arg_output)
 
    } //end LSTM_Cell


    

    
    def main (args: Array[String]): Unit = {
        
        //data processing 
        //prepare memories to pass to Accel

        val argRegOut = ArgOut[T]
    
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


        val zero_init = (0::256){(j) => 0.to[T] }
        val zeros = DRAM[T](1, 256)
        setMem(zeros, zero_init)


        //for debugging: 
        // val result_one = DRAM[T](1, 256)
        // val result_two = DRAM[T](1, 256)
        // val result_three = DRAM[T](1, 256)
        // val result_four = DRAM[T](1, 256)
        // val state_dram = DRAM[T](1, 256)
        // val output_dram = DRAM[T](1, 256)
        // val prediction_dram = DRAM[T](1, 1)
        // val bias_output_dram = DRAM[T](1,1)

        Accel { 

            val bias_output_layer = SRAM[T](1, 1)
            bias_output_layer(0,0) = 0.10646543651819229125976562500000.to[T]

            //set memories
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
        

            //for loop to scroll through input window

            val state = SRAM[T](1, 256)
            state load zeros
            val output = SRAM[T](1, 256)
            output load zeros

            val input = SRAM[T](1, 1)

            Sequential.Foreach(7 by 1) { a =>
                input(0, 0) = window(a)

                val input_gate_matrix = element_add(element_add(matrix_mult(input, weights_input_gate), matrix_mult(output, weights_input_hidden)), bias_input)        
                val input_gate = element_sigmoid(input_gate_matrix)
                //for debugging: result_two store input_gate
          
                val forget_gate_matrix= element_add(element_add(matrix_mult(input, weights_forget_gate), matrix_mult(output, weights_forget_hidden)), bias_forget)
                val forget_gate = element_sigmoid(forget_gate_matrix)
                //for debugging: result_three store forget_gate
                    
                val output_gate_matrix = element_add(element_add(matrix_mult(input, weights_output_gate), matrix_mult(output, weights_output_hidden)), bias_output)
                val output_gate = element_sigmoid(output_gate_matrix)
                //for debuggin: esult_four store output_gate

                val memory_cell_matrix = element_add(element_add(matrix_mult(input, weights_memory_cell), matrix_mult(output, weights_memory_cell_hidden)), bias_memory_cell)
                val memory_cell = element_tanh(memory_cell_matrix)
                //for debugging: result_one store memory_cell

                LSTM_Cell(input_gate, forget_gate, output_gate, memory_cell, state, output)
                //for debugging: state_dram store state
                //for debugging: output_dram store output

            } //end Sequential

            val prediction_sram = SRAM[T](1, 1)
            val prediction = (matrix_mult(output, weights_output))(0,0) + bias_output_layer(0,0)

            argRegOut := prediction
            
        } //end Accel
        

        println("prediction: " + argRegOut.value)

        //for debugging: 
        // printMatrix(getMatrix(result_one), "result_one: ")
        // printMatrix(getMatrix(result_two), "result_two: ")
        // printMatrix(getMatrix(result_three), "result_three: ")
        // printMatrix(getMatrix(result_four), "result_four: ")
        // printMatrix(getMatrix(state_dram), "state: ")
        // printMatrix(getMatrix(output_dram), "output: ")

    } //end main
}
