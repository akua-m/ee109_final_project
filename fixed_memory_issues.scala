import spatial.dsl._


@spatial object RestructuredTest2 extends SpatialApp { 

    // type T = Float  //Float = FltPt[_24, _8] = [_significant bits, _exponent bits]
    type T = FixPt[TRUE, _16, _16]


    /*
    def sigmoid(a: T) = { //: T = {
      // TODO: Fill in your sigmoid implementation.
      a

    }
    */

    /*
    def tanh(a: T) = {
      // TODO: Fill in your tanh implementation.
      a
    }
    */
    
    def matrix_mult(a_operand: SRAM2[T], b_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 

        MemReduce(store_matrix)(a_operand.cols by 1) { c => 
            val temp = SRAM[T](a_operand.rows, b_operand.cols)

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


    def element_sigmoid(a_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 

        //sigmoid formula: sigmoid(x) = (tanh(x/2) + 1) / 2

        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                val tanh_value = tanh((a_operand(r, c)) / 2)
                val sigmoid = ((tanh_value + 1) / 2)
                store_matrix(r, c) = sigmoid
            }
        }

        store_matrix

    } //end element_sigmoid


    def element_tanh(a_operand: SRAM2[T], store_matrix: SRAM2[T]) : SRAM2[T] = { 
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = tanh(a_operand(r, c))
            }
        }
        
        store_matrix
        
    } //end element_tanh

    def store_result (resultant: SRAM2[T], store_matrix: SRAM2[T]) : Unit = {
        
        Foreach(resultant.rows.to[Int] by 1) { r =>
            Foreach(resultant.cols.to[Int] by 1) { c =>
                store_matrix(r, c) = resultant(r, c)
            }
        }
    }


    def LSTM_Cell (arg_input_gate: SRAM2[T], arg_forget_gate: SRAM2[T], arg_output_gate: SRAM2[T], arg_memory_cell: SRAM2[T], arg_state: SRAM2[T], arg_output: SRAM2[T], state_mem: SRAM2[T], output_mem: SRAM2[T], store_matrix: SRAM2[T]): Unit =  {
        //state = state * forget_gate + input_gate * memory_cell    
        //output = output_gate * tf.tanh(state)
        
        val state = element_add(element_mult(arg_state, arg_forget_gate, state_mem(0::0, 0::256)), element_mult(arg_input_gate, arg_memory_cell, state_mem(1::1, 0::256)), state_mem(2::2, 0::256))
        store_result(state, arg_state)
        
        val output = element_mult(arg_output_gate, element_tanh(state, store_matrix), output_mem)
        store_result(output, arg_output)

    } //end LSTM_Cell


    def tanh(a_operand: T): T = {
        val sigmoid_lut = LUT[T](15,3)(0.390625.to[T], 0.453125.to[T], 0.4049.to[T], 0.453125.to[T], 0.515625.to[T], 0.4558.to[T], 0.515625.to[T], 0.578125.to[T], 0.5038.to[T], 0.578125.to[T], 0.640625.to[T], 0.5490.to[T], 0.640625.to[T], 0.703125.to[T], 0.5911.to[T], 0.703125.to[T], 0.78125.to[T], 0.6348.to[T], 0.78125.to[T], 0.859375.to[T], 0.6791.to[T], 0.859375.to[T], 0.9375.to[T], 0.7190.to[T], 0.9375.to[T], 1.046875.to[T], 0.7609.to[T], 1.046875.to[T], 1.171875.to[T], 0.8057.to[T], 1.171875.to[T], 1.328125.to[T], 0.8493.to[T], 1.328125.to[T], 1.53125.to[T], 0.8916.to[T], 1.53125.to[T], 1.859375.to[T], 0.9329.to[T], 1.859375.to[T], 2.90625.to[T], 0.9740.to[T], 2.90625.to[T], 2.90625.to[T], 1.to[T])
        val a_tanh = Reg[T](0)

        val temp = Reg[T](0)
        val temp2 = Reg[T](0)

        Sequential { 

            if (a_operand < 0) {
                temp := -1 * a_operand
            } else {
                temp := a_operand
            }
                

            if (sigmoid_lut(0,0) < temp && temp <= sigmoid_lut(0,1)) {
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
                a_tanh := 1 - temp2.value
            } else {
                a_tanh := temp2.value
            }
        
        } //end Sequential


        a_tanh.value  

    } //end tanh
    

    
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
        val weights_o_mem = DRAM[T](1, 256)
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

        val zero_init = (0::0, 0::256){(i,j) => 0.to[T] }
        val zeros = DRAM[T](1, 256)
        setMem(zeros, zero_init)

        val v = ArgIn[T]
        setArg(v, args(0).to[T])

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

            val weights_output = SRAM[T](1, 256)
            weights_output load weights_o_mem

            val weights_output_gate = SRAM[T](1, 256)
            weights_output_gate load weights_og_mem

            val weights_output_hidden = SRAM[T](256, 256)
            weights_output_hidden load weights_oh_mem

            val window = SRAM[T](7)
            window load window_mem
        

            //for loop to scroll through input window

            val state = SRAM[T](1, 256)
            val output = SRAM[T](1, 256)
            output load zeros

            val input = SRAM[T](1, 1)
            input(0,0) = v.value


            val input_sram = SRAM[T](4, 256)
            val forget_sram = SRAM[T](4, 256)
            val output_sram = SRAM[T](4, 256)
            val memory_sram = SRAM[T](4, 256)

            val state_sram = SRAM[T](3, 256)
            val output_lstm_sram = SRAM[T](1, 256)

            val activation_sram = SRAM[T](5, 256)



            Sequential.Foreach(7 by 1) { a =>

                //input = window(i)
                
                input(0, 0) = window(a)

                //4 gate calculations

                val input_gate_matrix = element_add(element_add(matrix_mult(weights_input_gate, input, input_sram(0::0, 0::256)), matrix_mult(output, weights_input_hidden, input_sram(1::1, 0::256)), input_sram(2::2, 0::256)), bias_input, input_sram(3::3, 0::256))        
                val input_gate = element_sigmoid(input_gate_matrix, activation_sram(0::0, 0::256))
      
                val forget_gate_matrix= element_add(element_add(matrix_mult(weights_forget_gate, input, forget_sram(0::0, 0::256)), matrix_mult(output, weights_forget_hidden, forget_sram(1::1, 0::256)), forget_sram(2::2, 0::256)), bias_forget, forget_sram(3::3, 0::256))
                val forget_gate = element_sigmoid(forget_gate_matrix, activation_sram(1::1, 0::256))
                
                val output_gate_matrix = element_add(element_add(matrix_mult(weights_output_gate, input, output_sram(0::0, 0::256)), matrix_mult(output, weights_output_hidden, output_sram(1::1, 0::256)), output_sram(2::2, 0::256)), bias_output, output_sram(3::3, 0::256))
                val output_gate = element_sigmoid(output_gate_matrix, activation_sram(2::2, 0::256))

                val memory_cell_matrix = element_add(element_add(matrix_mult(weights_memory_cell, input, memory_sram(0::0, 0::256)), matrix_mult(output, weights_memory_cell_hidden, output_sram(1::1, 0::256)), memory_sram(2::2, 0::256)), bias_memory_cell, memory_sram(3::3, 0::256))
                val memory_cell = element_tanh(memory_cell_matrix, activation_sram(3::3, 0::256))

                //dram = LSMT cell call 

                LSTM_Cell(input_gate, forget_gate, output_gate, memory_cell, state, output, state_sram, output_lstm_sram,activation_sram(4::4, 0::256))

            }                  
                    
            //end for loop

            //use last state and ouput for prediction equation
            val prediction_sram = SRAM[T](2, 256)
            val prediction = element_add(matrix_mult(output, weights_output, prediction_sram(0::0, 0::256)), bias_output_layer, prediction_sram(1::1, 0::256))

            argRegOut := prediction(0,0)

        }

        //println("prediction: " + argRegOut)
    }
}
