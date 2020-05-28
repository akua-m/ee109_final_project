
import spatial.dsl._


@spatial object RestructuredTest extends SpatialApp { 

	// type T = Float  //Float = FltPt[_24, _8] = [_significant bits, _exponent bits]
    type T = FixPt[TRUE, _16, _16]

    def sigmoid(a: T) = {
      // TODO: Fill in your sigmoid implementation.
      a
    }

    def tanh(a: T) = {
      // TODO: Fill in your tanh implementation.
      a
    }
	
	def matrix_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows.to[Int], b_operand.cols.to[Int])

        MemReduce(c_sram)(a_operand.cols by 1) { c => 
            val temp = SRAM[T](a_operand.rows, b_operand.cols)

            Foreach(a_operand.rows.to[Int] by 1) { a =>
                Foreach(b_operand.cols.to[Int] by 1) { b =>
                    temp(a, b) = a_operand(a, c) * b_operand(c, b)
                }
            }

            temp
        }{_+_}

	}


    def element_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = {  

		val c_sram = SRAM[T](a_operand.rows, a_operand.cols)  //a_operand and b_operand are the same size
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = a_operand(r, c) * b_operand(r, c)
            }
        }

        c_sram
	}

    def element_add(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

		val c_sram = SRAM[T](a_operand.rows, a_operand.cols)  //a_operand and b_operand are the same size
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                c_sram(r, c) = a_operand(r, c) + b_operand(r, c)
            }
        }

        c_sram
    }

    def element_sigmoid(a_operand: SRAM2[T]) : SRAM2[T] = { 

		val a_sigmoid = SRAM[T](a_operand.rows, a_operand.cols)  //a_operand and b_operand are the same size
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                a_sigmoid(r, c) = sigmoid(a_operand(r, c))
            }
        }

        a_sigmoid
     }

     def element_tanh(a_operand: SRAM2[T]) : SRAM2[T] = { 

		val a_tanh = SRAM[T](a_operand.rows, a_operand.cols)  //a_operand and b_operand are the same size
        
        Foreach(a_operand.rows.to[Int] by 1) { r =>
            Foreach(a_operand.cols.to[Int] by 1) { c =>
                a_tanh(r, c) = tanh(a_operand(r, c))
            }
        }

        a_tanh
      }

     def LSTM_Cell (arg_input_gate: SRAM2[T], arg_forget_gate: SRAM2[T], arg_output_gate: SRAM2[T], arg_memory_cell: SRAM2[T], arg_output: SRAM2[T], arg_state: SRAM2[T]): Unit =  {
		//state = state * forget_gate + input_gate * memory_cell	
		//output = output_gate * tf.tanh(state)
        
		val state = element_add(element_mult(arg_state, arg_forget_gate), element_mult(arg_input_gate, arg_memory_cell))

		val output = element_mult(arg_output_gate, element_tanh(state))

      }

	
     def main (args: Array[String]): Unit = {
        //data processing 
		//prepare memories to pass to Accel
		
        //val file_loc = "/home/jluquin/spatial/apps/src/"
        //val bias_forget_file = io.Source.fromFile(file_loc + "bias_forget.csv")

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

        //val initialize_data = Array.tabulate[T](256){i => 0.to[T]}
        val c_init = (0::0, 0::256){(i,j) => 0.to[T] }
        val d1 = DRAM[T](1, 256)
        setMem(d1, c_init)

        val v = ArgIn[T]
        setArg(v, args(0).to[T])

	Accel { 

        val bias_output_layer = SRAM[T](1, 1)
        //val input = args(0.10646543651819229125976562500000).to[Float]
        bias_output_layer(0,0) = 0.10646543651819229125976562500000.to[T]
        //val bias_olayer_array = loadCSV1D[T](s"jluquin@tucson.stanford.edu:/home/jluquin/spatial/apps/src/bias_output_layer_final.csv", " ")
        //val bias_ol_mem = DRAM[T](1)
        //setMem(bias_ol_mem, bias_olayer_array)
        //bias_output_layer load input

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
            output load d1

            val input = SRAM[T](1, 1)
            input(0,0) = v.value


            //val len = 1
            //val vec1 = Array.tabulate[T](len){i => i.to[T]}

            Foreach(7 by 1) { a =>

                //input = window(i)
                
                input(0, 0) = window(a)

                //4 gate calculations

                val input_gate_matrix = element_add(element_add(matrix_mult(input, weights_input_gate), matrix_mult(output, weights_input_hidden)), bias_input)        
                val input_gate = element_sigmoid(input_gate_matrix)
      
                val forget_gate_matrix= element_add(element_add(matrix_mult(input, weights_forget_gate), matrix_mult(output, weights_forget_hidden)), bias_forget)
                val forget_gate = element_sigmoid(forget_gate_matrix)
                
                val output_gate_matrix = element_add(element_add(matrix_mult(input, weights_output_gate), matrix_mult(output, weights_output_hidden)), bias_output)
                val output_gate = element_sigmoid(output_gate_matrix)

                val memory_cell_matrix = element_add(element_add(matrix_mult(input, weights_memory_cell), matrix_mult(output, weights_memory_cell_hidden)), bias_memory_cell)
                val memory_cell = element_tanh(memory_cell_matrix)

                //dram = LSMT cell call 

                LSTM_Cell(input_gate, forget_gate, output_gate, memory_cell, state, output)

            }                  
                    
			//end for loop

			//use last state and ouput for prediction equation
            val prediction = element_add(matrix_mult(output, weights_output), bias_output_layer)

            argRegOut := prediction(0,0)

		}

        //println("prediction: " + argRegOut)
	}
}