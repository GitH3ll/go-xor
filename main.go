package main

import (
	"fmt"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	fmt.Printf("Input: %v\n", xor.Input.RawMatrix().Data)
	fmt.Printf("Target: %v\n", xor.Target.RawMatrix().Data)

	xor.Train(100000, 0.1)

	fmt.Printf("Predicted for 0 0: %v\n", xor.ForwardProp([]float64{0, 0}).RawMatrix().Data)
	fmt.Printf("Predicted for 1 0: %v\n", xor.ForwardProp([]float64{1, 0}).RawMatrix().Data)
	fmt.Printf("Predicted for 0 1: %v\n", xor.ForwardProp([]float64{0, 1}).RawMatrix().Data)
	fmt.Printf("Predicted for 1 1: %v\n", xor.ForwardProp([]float64{1, 1}).RawMatrix().Data)
}
