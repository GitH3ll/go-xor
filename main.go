package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	xor.Train(100000, 0.1)
	fmt.Printf("Input: %v\n", xor.Input.RawMatrix().Data)
	fmt.Printf("Target: %v\n", xor.Target.RawMatrix().Data)
	fmt.Printf("Predicted: %v\n", xor.ForwardProp(mat.NewDense(1, 2, []float64{1, 0})).RawMatrix().Data)
}
