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
	Input            *mat.Dense
	Target           *mat.Dense
	HiddenW          *mat.Dense
	HiddenB          *mat.Dense
	HiddenErrorW     *mat.Dense
	HiddenOutput     *mat.Dense
	DerivedHidden    *mat.Dense
	OutW             *mat.Dense
	OutB             *mat.Dense
	Predicted        *mat.Dense
	DerivedPredicted *mat.Dense
	Error            *mat.Dense
}

func NewXor() *Xor {
	rand.Seed(time.Now().UnixNano())
	return &Xor{
		Input: mat.NewDense(4, inputLayerIs2,
			[]float64{0, 0, 0, 1, 1, 0, 1, 1}),

		Target: mat.NewDense(4, 1,
			[]float64{0, 1, 1, 0}),

		HiddenW: mat.NewDense(inputLayerIs2, hiddenLayerIs2,
			[]float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}),

		HiddenB: mat.NewDense(1, hiddenLayerIs2,
			[]float64{rand.Float64(), rand.Float64()}),

		HiddenErrorW: &mat.Dense{},

		DerivedHidden: &mat.Dense{},

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

func (x *Xor) Train(epochs int, lr float64) {
	for i := 0; i < epochs; i++ {
		x.ForwardProp()
		x.BackProp()
		x.UpdateWeights(lr)
	}
}

func (x *Xor) ForwardProp() *mat.Dense {
	var hiddenActivation mat.Dense
	hiddenActivation.Mul(x.Input, x.HiddenW)
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

	return mat.DenseCopyOf(x.Predicted)
}

func (x *Xor) BackProp() {
	x.Error.Apply(func(i, j int, v float64) float64 {
		return squareError(x.Target.At(i, j), x.Predicted.At(i, j))
	}, x.Error)

	temp := mat.DenseCopyOf(x.Predicted)
	temp.Apply(sigmoidDerivativeMat, temp)
	x.DerivedPredicted.MulElem(x.Error, temp)

	x.HiddenErrorW.Mul(x.DerivedPredicted, x.OutW.T())

	temp = mat.DenseCopyOf(x.HiddenOutput)
	temp.Apply(sigmoidDerivativeMat, temp)
	x.DerivedHidden.MulElem(x.HiddenErrorW, temp)
}

func (x *Xor) UpdateWeights(lr float64) {
	temp := mat.DenseCopyOf(x.HiddenOutput.T())
	var r mat.Dense
	r.Mul(temp, x.DerivedPredicted)
	r.Scale(lr, &r)
	x.OutW.Add(x.OutW, &r)

	x.OutB.Apply(func(i, j int, v float64) float64 {
		return v + mat.Sum(x.DerivedPredicted)*lr
	}, x.OutB)

	temp = mat.DenseCopyOf(x.Input.T())
	r.Reset()
	r.Mul(temp, x.DerivedHidden)
	r.Scale(lr, &r)
	x.HiddenW.Add(x.HiddenW, &r)

	x.HiddenB.Apply(func(i, j int, v float64) float64 {
		return v + mat.Sum(x.DerivedHidden)*lr
	}, x.HiddenB)
}
