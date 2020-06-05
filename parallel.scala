import spatial.dsl._


@spatial object LSTM extends SpatialApp { 

    type T = FixPt[TRUE, _16, _16]

    
    def matrix_mult(a_operand: SRAM2[T], b_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 

        MemReduce(store_matrix)(a_operand.cols by 1) { c => 
            val temp = SRAM[T](store_matrix.rows, store_matrix.cols)

            Foreach(a_operand.rows.to[Int] by 1) { a =>
                Foreach(b_operand.cols.to[Int] by 1) { b =>
                    temp(a, b) = a_operand(a, c) * b_operand(c, b)
                }
            }

            temp
        }{_+_}

    } //end matrix_mult


    def element_mult(a_operand: SRAM2[T], b_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = {  
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = a_operand(r, c) * b_operand(r, c)
            }
        }

        store_matrix

    } //end element_mult


    def element_add(a_operand: SRAM2[T], b_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = a_operand(r, c) + b_operand(r, c)
            }
        }

        store_matrix

    } //end element_add


    // def element_sigmoid(a_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 

    //     //sigmoid formula: sigmoid(x) = (tanh(x/2) + 1) / 2

    //     Foreach(a_operand.rows.to[Int] by 1) { r =>
    //         Foreach(a_operand.cols.to[Int] by 1) { c =>
    //             //val tanh_value = tanh((a_operand(r, c) / 2))
    //             //val sigmoid = (tanh_value + 1) / 2
    //             if (a_operand(r,c) < -2.5){
    //                 store_matrix(r,c) = 0
    //             } else if (2.5 < a_operand(r,c)) {
    //                 store_matrix(r,c) = 1
    //             } else {
    //                 store_matrix(r,c) = 0.2 * a_operand(r,c) + 0.5
    //             }
    //         }
    //     }

    //     store_matrix

    // } //end element_sigmoid

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

    def element_sigmoid(a_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 

        //sigmoid formula: sigmoid(x) = (tanh(x/2) + 1) / 2

        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                store_matrix(r,c) = sigmoid(a_operand(r,c))
            }
        }

        store_matrix

    } //end element_sigmoid


    def element_tanh(a_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = 2 * sigmoid(2 * a_operand(r,c)) - 1
            }
        }
        
        store_matrix
        
    } //end element_tanh

    // def element_tanh(a_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 
        
    //     Foreach(a_operand.rows.to[Int] by 1) { r =>
    //         Foreach(a_operand.cols.to[Int] by 1) { c =>
    //             store_matrix(r, c) = tanh(a_operand(r, c))
    //         }
    //     }
        
    //     store_matrix
        
    // } //end element_tanh

    // def tanh(a_operand: T): T = {
    //     val sigmoid_lut = LUT[T](15,3)(0.390625.to[T], 0.453125.to[T], 0.4049.to[T], 
    //                                     0.453125.to[T], 0.515625.to[T], 0.4558.to[T], 
    //                                     0.515625.to[T], 0.578125.to[T], 0.5038.to[T], 
    //                                     0.578125.to[T], 0.640625.to[T], 0.5490.to[T], 
    //                                     0.640625.to[T], 0.703125.to[T], 0.5911.to[T], 
    //                                     0.703125.to[T], 0.78125.to[T], 0.6348.to[T], 
    //                                     0.78125.to[T], 0.859375.to[T], 0.6791.to[T], 
    //                                     0.859375.to[T], 0.9375.to[T], 0.7190.to[T], 
    //                                     0.9375.to[T], 1.046875.to[T], 0.7609.to[T], 
    //                                     1.046875.to[T], 1.171875.to[T], 0.8057.to[T], 
    //                                     1.171875.to[T], 1.328125.to[T], 0.8493.to[T], 
    //                                     1.328125.to[T], 1.53125.to[T], 0.8916.to[T], 
    //                                     1.53125.to[T], 1.859375.to[T], 0.9329.to[T], 
    //                                     1.859375.to[T], 2.90625.to[T], 0.9740.to[T], 
    //                                     2.90625.to[T], 2.90625.to[T], 1.to[T])

    //     val a_tanh = Reg[T](0)

    //     val temp = Reg[T](0)
    //     val temp2 = Reg[T](0)

    //     Sequential { 

    //         if (a_operand < 0) {
    //             temp := -1 * a_operand
    //         } else {
    //             temp := a_operand
    //         }
                

    //         if (temp <= sigmoid_lut(0,0)){ 
    //             temp2 := temp
    //         } else if (sigmoid_lut(0,0) < temp && temp <= sigmoid_lut(0,1)) {
    //             temp2 := sigmoid_lut(0,2) 
    //         } else if (sigmoid_lut(1,0) < temp && temp <= sigmoid_lut(1,1)) {
    //             temp2 := sigmoid_lut(1,2) 
    //         } else if (sigmoid_lut(2,0) < temp && temp <= sigmoid_lut(2,1)) {
    //             temp2 := sigmoid_lut(2,2)
    //         } else if (sigmoid_lut(3,0) < temp && temp <= sigmoid_lut(3,1)) {
    //             temp2 := sigmoid_lut(3,2)
    //         } else if (sigmoid_lut(4,0) < temp && temp <= sigmoid_lut(4,1)) {
    //             temp2 := sigmoid_lut(4,2)
    //         } else if (sigmoid_lut(5,0) < temp && temp <= sigmoid_lut(5,1)) {
    //             temp2 := sigmoid_lut(5,2)
    //         } else if (sigmoid_lut(6,0) < temp && temp <= sigmoid_lut(6,1)) {
    //             temp2 := sigmoid_lut(6,2)
    //         } else if (sigmoid_lut(7,0) < temp && temp <= sigmoid_lut(7,1)) {
    //             temp2 := sigmoid_lut(7,2)
    //         } else if (sigmoid_lut(8,0) < temp && temp <= sigmoid_lut(8,1)) {
    //             temp2 := sigmoid_lut(8,2)
    //         } else if (sigmoid_lut(9,0) < temp && temp <= sigmoid_lut(9,1)) {
    //             temp2 := sigmoid_lut(9,2)
    //         } else if (sigmoid_lut(10,0) < temp && temp <= sigmoid_lut(10,1)) {
    //             temp2 := sigmoid_lut(10,2)
    //         } else if (sigmoid_lut(11,0) < temp && temp<= sigmoid_lut(11,1)) {
    //             temp2 := sigmoid_lut(11,2)
    //         } else if (sigmoid_lut(12,0) < temp && temp<= sigmoid_lut(12,1)) {
    //             temp2 := sigmoid_lut(12,2)
    //         } else if (sigmoid_lut(13,0) < temp && temp <= sigmoid_lut(13,1)) {
    //             temp2 := sigmoid_lut(13,2)
    //         } else {
    //             temp2 := sigmoid_lut(14,2)
    //         }
            

    //         if (a_operand < 0) {
    //             a_tanh := -1 * temp2.value
    //         } else {
    //             a_tanh := temp2.value
    //         }
        
    //     } //end Sequential

    //     a_tanh.value  

    // } //end tanh


    def store_result (resultant: SRAM2[T], store_matrix: SRAM2[T]) : Unit = {
        
        Foreach(resultant.rows.to[Int] by 1) { r =>
            Foreach(resultant.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = resultant(r, c)
            }
        }
    } //end store_result


    def LSTM_Cell (arg_input_gate: SRAM2[T], arg_forget_gate: SRAM2[T], arg_output_gate: SRAM2[T], arg_memory_cell: SRAM2[T], arg_state: SRAM2[T], arg_output: SRAM2[T], state_mem1: SRAM2[T], state_mem2: SRAM2[T], state_mem3: SRAM2[T], output_mem: SRAM2[T], store_matrix: SRAM2[T]): Unit =  {
        //state = state * forget_gate + input_gate * memory_cell   
        //output = output_gate * tf.tanh(state)
       
        val state = element_add(element_mult(arg_state, arg_forget_gate, state_mem1), element_mult(arg_input_gate, arg_memory_cell, state_mem2), state_mem3)
        store_result(state, arg_state)
       
        val output = element_mult(arg_output_gate, element_tanh(state, store_matrix), output_mem)
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
        //val result_one = DRAM[T](1, 256)
        //val result_two = DRAM[T](1, 256)
        //val result_three = DRAM[T](1, 256)
        //val result_four = DRAM[T](1, 256)
        //val state_dram = DRAM[T](1, 256)
        //val output_dram = DRAM[T](1, 256)
        //val prediction_dram = DRAM[T](1, 1)
        //val bias_output_dram = DRAM[T](1,1)

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

            val input_sram1 = SRAM[T](1, 256)
            val input_sram2 = SRAM[T](1, 256)
            val input_sram3 = SRAM[T](1, 256)
            val input_sram4 = SRAM[T](1, 256)

            val forget_sram1 = SRAM[T](1, 256)
            val forget_sram2 = SRAM[T](1, 256)
            val forget_sram3 = SRAM[T](1, 256)
            val forget_sram4 = SRAM[T](1, 256)

            val output_sram1 = SRAM[T](1, 256)
            val output_sram2 = SRAM[T](1, 256)
            val output_sram3 = SRAM[T](1, 256)
            val output_sram4 = SRAM[T](1, 256)

            val memory_sram1 = SRAM[T](1, 256)
            val memory_sram2 = SRAM[T](1, 256)
            val memory_sram3 = SRAM[T](1, 256)
            val memory_sram4 = SRAM[T](1, 256)

            val activation_sram1 = SRAM[T](1, 256)
            val activation_sram2 = SRAM[T](1, 256)
            val activation_sram3 = SRAM[T](1, 256)
            val activation_sram4 = SRAM[T](1, 256)
            val activation_sram5 = SRAM[T](1, 256)

            val state_sram1 = SRAM[T](1, 256)
            val state_sram2 = SRAM[T](1, 256)
            val state_sram3 = SRAM[T](1, 256)

            val input_gate = SRAM[T](1, 256)
            val forget_gate = SRAM[T](1, 256)
            val output_gate = SRAM[T](1, 256)
            val memory_cell = SRAM[T](1, 256)

            val output_lstm_sram = SRAM[T](1, 256)


            Sequential.Foreach(7 by 1) { a =>
                input(0, 0) = window(a)
                Parallel {
                    val input_gate_matrix = element_add(element_add(matrix_mult(input, weights_input_gate, input_sram1), matrix_mult(output, weights_input_hidden, input_sram2), input_sram3), bias_input, input_sram4)        
                    val input_gate1 = element_sigmoid(input_gate_matrix, activation_sram2)
                    //for debugging: result_two store input_gate
              
                    val forget_gate_matrix= element_add(element_add(matrix_mult(input, weights_forget_gate, forget_sram1), matrix_mult(output, weights_forget_hidden, forget_sram2), forget_sram3), bias_forget, forget_sram4)
                    val forget_gate1 = element_sigmoid(forget_gate_matrix, activation_sram3)
                    //for debugging: result_three store forget_gate
                        
                    val output_gate_matrix = element_add(element_add(matrix_mult(input, weights_output_gate, output_sram1), matrix_mult(output, weights_output_hidden, output_sram2), output_sram3), bias_output, output_sram4)
                    val output_gate1 = element_sigmoid(output_gate_matrix, activation_sram4)
                    //for debuggin: result_four store output_gate

                    val memory_cell_matrix = element_add(element_add(matrix_mult(input, weights_memory_cell, memory_sram1), matrix_mult(output, weights_memory_cell_hidden, memory_sram2), memory_sram3), bias_memory_cell, memory_sram4)
                    val memory_cell1 = element_tanh(memory_cell_matrix, activation_sram1)
                    //for debugging: result_one store memory_cell

                    store_result(input_gate1, input_gate)
                    store_result(forget_gate1, forget_gate)
                    store_result(output_gate1, output_gate)
                    store_result(memory_cell1, memory_cell)
                }
                


                LSTM_Cell(input_gate, forget_gate, output_gate, memory_cell, state, output, state_sram1, state_sram2, state_sram3, output_lstm_sram, activation_sram5)
                //for debugging: state_dram store state
                //for debugging: output_dram store output

            } //end Sequential

            val prediction_sram = SRAM[T](1, 1)
            val prediction = (matrix_mult(output, weights_output, prediction_sram))(0,0) + bias_output_layer(0,0)

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
