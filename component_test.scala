import spatial.dsl._


@spatial object test extends SpatialApp { 

    type T = FixPt[TRUE, _16, _16]

    
    def matrix_mult(a_operand: SRAM2[T], b_operand: SRAM2[T]) : SRAM2[T] = { 

        val c_sram = SRAM[T](a_operand.rows, b_operand.cols).buffer

        val tileM = 16
        val tileN = 16
        val tileK = 16


        Foreach(a_operand.cols by tileK){kk =>
            val numel_k = min(tileK.to[Int], a_operand.cols - kk)
            Foreach(a_operand.rows by tileM){mm =>
                val numel_m = min(tileM.to[Int], a_operand.rows - mm)
                //val tileA = SRAM[T](16, 16)
                //store_result(a_operand(mm::mm+numel_m, kk::kk+numel_k), tileA)
                Foreach(b_operand.cols by tileN){nn =>
                    val numel_n = min(tileN.to[Int], b_operand.cols - nn)
                    //val tileB = SRAM[T](16, 16)
                    //store_result(b_operand(kk::kk+numel_k, nn::nn+numel_n), tileB)
                    val tileC = SRAM[T](16, 16).buffer
                    store_result(c_sram(mm::mm+numel_m, nn::nn+numel_n), tileC)

                    // Your code here
                    MemFold(tileC)(numel_k by 1) { k =>
                        val temp = SRAM[T](tileM, tileN)

                        Foreach(numel_m by 1) { m =>
                          Foreach(numel_n by 1) { n =>
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

    def init (input: SRAM2[T]) : Unit = {
        
        Foreach(input.rows.to[Int] by 1) { r =>
            Foreach(input.cols.to[Int] by 1) { c =>
                input(r, c) = 0.to[T]
            }
        }
    } //end init


    def LSTM_Cell (arg_input_gate: SRAM2[T], arg_forget_gate: SRAM2[T], arg_output_gate: SRAM2[T], arg_memory_cell: SRAM2[T], arg_state: SRAM2[T], arg_output: SRAM2[T]): Unit =  {
        //state = state * forget_gate + input_gate * memory_cell   
        //output = output_gate * tf.tanh(state)
       
        val state = element_add(element_mult(arg_state, arg_forget_gate), element_mult(arg_input_gate, arg_memory_cell))
        store_result(state, arg_state)
       
        val output = element_mult(arg_output_gate, element_tanh(state))
        store_result(output, arg_output)
 
    } //end LSTM_Cell


    

    
    def main (args: Array[String]): Unit = {

        val sram = SRAM[T](4, 4)

        sram(0,0) = 5
        sram(0,1) = 10
        sram(1,0) = 15
        sram(1,1) = 20

        sram(0,2) = 100
        sram(0,3) = 200
        sram(1,2) = 300
        sram(1,3) = 400

        sram(2,0) = 500
        sram(2,1) = 600
        sram(2,2) = 700
        sram(2,3) = 800

        sram(3,0) = 150
        sram(3,1) = 250
        sram(3,2) = 350
        sram(3,3) = 450


        val a_operand = SRAM[T](1, 1)
        a_operand(0,0) = 1
        //a_operand(0,1) = 1
        //a_operand(1,0) = 1
        //a_operand(1,1) = 1

        val b_operand = SRAM[T](1, 3)
        b_operand(0,0) = 2
        b_operand(0,1) = 2
        b_operand(0,2) = 2
        //b_operand(1,0) = 2
        //b_operand(1,1) = 2
        //b_operand(1,2) = 2


        //store_result(sram(0::2, 0::2), store)


        val c_sram = SRAM[T](a_operand.rows, b_operand.cols).buffer
        init(c_sram)
        

        val tileM = 16
        val tileN = 16
        val tileK = 16


        Foreach(a_operand.cols by tileK){kk =>
            val numel_k = min(tileK.to[Int], a_operand.cols - kk)
            Foreach(a_operand.rows by tileM){mm =>
                val numel_m = min(tileM.to[Int], a_operand.rows - mm)
                //val tileA = SRAM[T](16, 16)
                //store_result(a_operand(mm::mm+numel_m, kk::kk+numel_k), tileA)
                Foreach(b_operand.cols by tileN){nn =>
                    val numel_n = min(tileN.to[Int], b_operand.cols - nn)
                    //val tileB = SRAM[T](16, 16)
                    //store_result(b_operand(kk::kk+numel_k, nn::nn+numel_n), tileB)
                    val tileC = SRAM[T](16, 16).buffer
                    store_result(c_sram(mm::mm+numel_m, nn::nn+numel_n), tileC)

                    // Your code here
                    MemFold(tileC)(numel_k by 1) { k =>
                        val temp = SRAM[T](tileM, tileN)

                        Foreach(numel_m by 1) { m =>
                          Foreach(numel_n by 1) { n =>
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


        println("(0,0): " + c_sram(0,0))
        println("(0,1): " + c_sram(0,1))
        println("(1,0): " + c_sram(0,2))
        //println("(1,1): " + store(1,1))

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
