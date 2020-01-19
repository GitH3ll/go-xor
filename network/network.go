package network

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

const inputLayerIs2 = 2
const hiddenLayerIs2 = 2
const outputLayerIs1 = 1

type Xor struct {
	HiddenW mat.Matrix
	HiddenB mat.Matrix
	OutW    mat.Matrix
	OutB    mat.Matrix
	Data    [][][]float64
}

func NewXor() *Xor {
	rand.Seed(time.Now().UnixNano())
	return &Xor{
		HiddenW: mat.NewDense(inputLayerIs2, hiddenLayerIs2,
			[]float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}),
		HiddenB: mat.NewDense(1, hiddenLayerIs2,
			[]float64{rand.Float64(), rand.Float64()}),
		OutW: mat.NewDense(hiddenLayerIs2, outputLayerIs1,
			[]float64{rand.Float64(), rand.Float64()}),
		OutB: mat.NewDense(1, outputLayerIs1,
			[]float64{rand.Float64()}),
		Data: [][][]float64{
			{{0, 0}, {0}},
			{{1, 0}, {1}},
			{{0, 1}, {1}},
			{{1, 1}, {0}},
		},
	}
}

func (x *Xor) ForwardProp(input []float64) float64 {
	inputMat := mat.NewDense(1, hiddenLayerIs2, input)
	hiddenActivation := mat.NewDense(1, hiddenLayerIs2, nil)
	hiddenActivation.Mul(inputMat, x.HiddenW)
	hiddenActivation.Add(x.HiddenB, hiddenActivation)
	hiddenOutput := mat.NewDense(1, hiddenLayerIs2, nil)
	hiddenOutput.Apply(sigmoidMat, hiddenActivation)

	outputActivation := mat.NewDense(1, outputLayerIs1, nil)
	outputActivation.Mul(hiddenOutput, x.OutW)
	outputActivation.Add(x.OutB, outputActivation)
	predicted := mat.NewDense(1, outputLayerIs1, nil)
	predicted.Apply(sigmoidMat, outputActivation)

	return predicted.At(0, 0)
}
