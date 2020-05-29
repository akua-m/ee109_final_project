import spatial.dsl._


@spatial object test extends SpatialApp { 

    // type T = Float  //Float = FltPt[_24, _8] = [_significant bits, _exponent bits]
    type T = FixPt[TRUE, _16, _16]

    
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

        val a = DRAM[T](2, 2)
        val b = DRAM[T](2, 2)
        val c = DRAM[T](2, 2)

        Accel { 

            val a_sram = SRAM[T](2, 2)
            a_sram(0,0) = 1
            a_sram(0,1) = 1
            a_sram(1,0) = 1
            a_sram(1,1) = 1

            val b_sram = SRAM[T](2, 2)
            b_sram(0,0) = 2
            b_sram(0,1) = 2
            b_sram(1,0) = 3
            b_sram(1,1) = 4

            val store = SRAM[T](2, 2)
            //val resulta = element_mult(a_sram, b_sram, store)
            //val resultb = element_add(a_sram, b_sram, store)
            val resultc = matrix_mult(a_sram, b_sram, store)

            a store resulta
            b store resultb
            c store resultc

        }

        //printMatrix(getMatrix(a), "element mult: ")
        //printMatrix(getMatrix(b), "element add: ")
        printMatrix(getMatrix(c), "matrix mult: ")

    }
}
