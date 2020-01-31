package main

import (
	"fmt"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	xor.Train(100000, 0.1)
	fmt.Printf("Input: %v\n", xor.Input.RawMatrix().Data)
	fmt.Printf("Target: %v\n", xor.Target.RawMatrix().Data)
	fmt.Printf("Predicted: %v\n", xor.ForwardProp().RawMatrix().Data)
}
