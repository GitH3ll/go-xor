package main

import (
	"fmt"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	xor.Train(100000, 0.1)
	fmt.Println(xor.ForwardProp().RawMatrix().Data)
	fmt.Println(xor.Target.RawMatrix().Data)
}
