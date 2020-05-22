import spatial.dsl._


@spatial object test extends SpatialApp { 

	type T = Float  //Float = FltPt[_24, _8] = [_significant bits, _exponent bits]
	
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

        val c_dram = DRAM[T](2, 2)
        val d_dram = DRAM[T](2, 2)
        val e_dram = DRAM[T](2, 2)
        val f_dram = DRAM[T](2, 2)
        val g_dram = DRAM[T](2, 2)
        val h_dram = DRAM[T](2, 2)

        val bias_forget_array = loadCSV1D[T](s"$DATA/Desktop/cvs/bias_forget_final.csv", " ")
        val bias_f_mem = DRAM[T](1, 256)
        setMem(bias_f_mem, bias_forget_array)

        Accel { 

            val a = SRAM[T](2,2)
            a(0,0) = 1
            a(0,1) = 1
            a(1,0) = 1
            a(1,1) = 1

            val b = SRAM[T](2,2)
            b(0,0) = 2
            b(0,1) = 2
            b(1,0) = 3
            b(1,1) = 3

            val c = matrix_mult(a, b)
            val d = element_mult(a, b)
            val e = element_add(a, b)
            val f = element_sigmoid(a)
            val g = element_tanh(a)

            val h = element_add(matrix_mult(a, b), element_tanh(a))


            c_dram store c
            d_dram store d
            e_dram store e
            f_dram store f
            g_dram store g
            h_dram store h

        }

        /*
        printMatrix(getMatrix(c_dram))
        printMatrix(getMatrix(d_dram))
        printMatrix(getMatrix(e_dram))
        printMatrix(getMatrix(f_dram))
        printMatrix(getMatrix(g_dram))
        printMatrix(getMatrix(h_dram))
        */

        printMatrix(getMatrix(bias_f_mem))

    }
}