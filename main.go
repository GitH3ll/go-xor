package main

import (
	"fmt"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	fmt.Println(xor.ForwardProp([]float64{0, 0}))
}
