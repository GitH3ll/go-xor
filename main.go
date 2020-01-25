package main

import (
	"fmt"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	fmt.Println(xor.ForwardProp(nil))
	xor.BackProp(0.1)
	xor.UpdateWeights()
}
