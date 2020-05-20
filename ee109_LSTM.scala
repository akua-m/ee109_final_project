import spatial.dsl._


@spatial object LSTMAccelerator extends SpatialApp { 

	type T = Float  //Float = FltPt[_24, _8] = [_significant bits, _exponent bits]
	
	def matrix_mult(a_operand: Matrix[T], b_operand: Matrix[T]) : Matrix[T] = { 

		val a_rows = ArgIn[Int]
   		val a_cols = ArgIn[Int]
   		val b_rows = ArgIn[Int]
   		val b_cols = ArgIn[Int]
   		setArg(a_rows, a_operand.rows)
   		setArg(a_cols, a_operand.cols)
   		setArg(b_rows, b_operand.rows)
   		setArg(b_cols, b_operand.cols)

		val a = DRAM[T](a_rows, a_cols)
		val b = DRAM[T](b_rows, b_cols)
		val c = DRAM[T](a_rows, b_cols)

		setMem(a, a_operand)
		setMem(b, b_operand)


		Accel { 
			val a_sram = SRAM[T](a_rows, a_cols)
			val b_sram = SRAM[T](b_rows, b_cols)
			val c_sram = SRAM[T](a_rows, b_cols)

            MemReduce(c_sram)(a_cols by 1) { i => 
                val temp = SRAM[T](a_rows, b_cols)

                Foreach(a_rows by 1) { m =>
                	Foreach(b_cols by 1) { n =>
                    	temp(m, n) = a_sram(m, i) * b_sram(i, n)
                  	}
                }

                temp
            }{_+_}

            c store c_sram
		}

		getMatrix(c)  //return value
	}

	def LSTM_Cell (arg_input_gate: Matrix[T], arg_forget_gate: Matrix[T], arg_output_gate: Matrix[T], arg_memory_cell: Matrix[T], arg_output: Matrix[T], arg_state: Matrix[T]) : DRAM[T](512) = {
		//state = state * forget_gate + input_gate * memory_cell	
		//output = output_gate * tf.tanh(state)

		val state = SRAM[T](256)
		val output = SRAM[T](256)

		val state_data = arg_state*arg_forget_gate + arg_input_gate*arg_memory_cell
		setMem(state, state_data)

		val output_data = arg_output_gate*tanh(state_data)
		setMem(output, output_data)

		val result = DRAM[T](512)
		result(0::256) store state
		result(256::512) store output

		result

	}

	def main {

		//process data
		//calculate: input_gate, forget_gate, output_gate, memory_cell

		//loop through window and call LSTMAccelerator

		//calculate prediction



	}


}