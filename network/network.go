package network

import (
	"log"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

const inputLayerIs2 = 2
const hiddenLayerIs2 = 2
const outputLayerIs1 = 1

type Xor struct {
	Input            *mat.Dense
	HiddenW          *mat.Dense
	HiddenB          *mat.Dense
	HiddenErrorW     *mat.Dense
	HiddenOutput     *mat.Dense
	DerivedHidden    *mat.Dense
	OutW             *mat.Dense
	OutB             *mat.Dense
	Predicted        *mat.Dense
	DerivedPredicted *mat.Dense
	Target           *mat.Dense
	Error            *mat.Dense
}

func NewXor() *Xor {
	rand.Seed(time.Now().UnixNano())
	return &Xor{
		Input: mat.NewDense(4, inputLayerIs2,
			[]float64{0, 0, 1, 1, 0, 1, 1, 0}),

		Target: mat.NewDense(4, 1,
			[]float64{0, 0, 1, 1}),

		HiddenW: mat.NewDense(inputLayerIs2, hiddenLayerIs2,
			[]float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}),

		HiddenB: mat.NewDense(1, hiddenLayerIs2,
			[]float64{rand.Float64(), rand.Float64()}),

		HiddenErrorW: mat.NewDense(1, 1, nil),

		DerivedHidden: mat.NewDense(1, 1, nil),

		HiddenOutput: mat.NewDense(4, hiddenLayerIs2, nil),

		OutW: mat.NewDense(hiddenLayerIs2, outputLayerIs1,
			[]float64{rand.Float64(), rand.Float64()}),

		OutB: mat.NewDense(1, outputLayerIs1,
			[]float64{rand.Float64()}),

		Predicted: mat.NewDense(4, 1, nil),

		DerivedPredicted: mat.NewDense(4, 1, nil),

		Error: mat.NewDense(4, 1, nil),
	}
}

func (x *Xor) ForwardProp(input *mat.Dense) mat.Dense {
	if input == nil {
		input = x.Input
	}

	var hiddenActivation mat.Dense
	hiddenActivation.Mul(input, x.HiddenW)
	hiddenActivation.Apply(func(i, j int, v float64) float64 {
		return v + x.HiddenB.At(0, 0)
	}, &hiddenActivation)
	x.HiddenOutput.Apply(sigmoidMat, &hiddenActivation)

	var outputActivation mat.Dense
	outputActivation.Mul(x.HiddenOutput, x.OutW)
	outputActivation.Apply(func(i, j int, v float64) float64 {
		return v + x.OutB.At(0, 0)
	}, &outputActivation)
	x.Predicted.Apply(sigmoidMat, &outputActivation)

	return *x.Predicted
}

func (x *Xor) BackProp(lr float64) {
	print(x.Target)

	x.Error.Apply(func(i, j int, v float64) float64 {
		return squareError(x.Target.At(i, j), x.Predicted.At(i, j))
	}, x.Error)

	temp := mat.DenseCopyOf(x.Predicted)
	temp.Apply(sigmoidDerivativeMat, temp)
	x.DerivedPredicted.MulElem(x.Error, temp)

	log.Println(x.Error)
	log.Println(x.Target)
}
func matrixSum(dense *mat.Dense) *mat.Dense {
	var sum float64
	for _, w := range dense.RawMatrix().Data {
		sum += w
	}
	return mat.NewDense(1, 1, []float64{sum})
}
