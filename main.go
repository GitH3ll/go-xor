package main

import (
	"log"

	"github.com/GitH3ll/go-xor/network"
)

func main() {
	xor := network.NewXor()
	xor.Train(100000, 0.05)
	log.Println(xor.ForwardProp().RawMatrix().Data)
}
