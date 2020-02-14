package network

import (
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

const inputLayer = 2
const hiddenLayer = 2
const outputLayer = 1
const trainSamples = 4

type Xor struct {
	Input            *mat.Dense
	Target           *mat.Dense
	HiddenW          *mat.Dense
	HiddenErrorW     *mat.Dense
	HiddenActivation *mat.Dense
	HiddenOutput     *mat.Dense
	DerivedHidden    *mat.Dense
	OutW             *mat.Dense
	OutputActivation *mat.Dense
	Predicted        *mat.Dense
	DerivedPredicted *mat.Dense
	Error            *mat.Dense
}

func NewXor() *Xor {
	rand.Seed(time.Now().UnixNano())
	return &Xor{
		Input: mat.NewDense(trainSamples, inputLayer,
			[]float64{0, 0, 0, 1, 1, 0, 1, 1}),

		Target: mat.NewDense(trainSamples, outputLayer,
			[]float64{0, 1, 1, 0}),

		HiddenW: mat.NewDense(inputLayer, hiddenLayer,
			[]float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}),

		HiddenErrorW: &mat.Dense{},

		HiddenActivation: &mat.Dense{},

		DerivedHidden: &mat.Dense{},

		HiddenOutput: mat.NewDense(trainSamples, hiddenLayer, nil),

		OutW: mat.NewDense(hiddenLayer, outputLayer,
			[]float64{rand.Float64(), rand.Float64()}),

		OutputActivation: &mat.Dense{},

		Predicted: mat.NewDense(trainSamples, outputLayer, nil),

		DerivedPredicted: mat.NewDense(trainSamples, outputLayer, nil),

		Error: mat.NewDense(trainSamples, outputLayer, nil),
	}
}

func (x *Xor) Train(epochs int, lr float64) {
	for i := 0; i < epochs; i++ {
		x.ForwardProp(nil)
		x.BackProp()
		x.UpdateWeights(lr)
	}
}

func (x *Xor) ForwardProp(signal []float64) *mat.Dense {
	if signal != nil {
		rows := len(signal) / inputLayer
		x.Input = mat.NewDense(rows, inputLayer, signal)
	}
	x.HiddenActivation.Reset()
	x.OutputActivation.Reset()
	x.HiddenOutput.Reset()
	x.Predicted.Reset()

	x.HiddenActivation.Mul(x.Input, x.HiddenW)
	x.HiddenOutput.Apply(sigmoidMat, x.HiddenActivation)

	x.OutputActivation.Mul(x.HiddenOutput, x.OutW)
	x.Predicted.Apply(sigmoidMat, x.OutputActivation)

	return mat.DenseCopyOf(x.Predicted)
}

func (x *Xor) BackProp() {
	x.Error.Apply(func(i, j int, v float64) float64 {
		return calculateError(x.Target.At(i, j), x.Predicted.At(i, j))
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
	temp := x.HiddenOutput.T()
	var r mat.Dense
	r.Mul(temp, x.DerivedPredicted)
	r.Scale(lr, &r)
	x.OutW.Add(x.OutW, &r)

	temp = x.Input.T()
	r.Reset()
	r.Mul(temp, x.DerivedHidden)
	r.Scale(lr, &r)
	x.HiddenW.Add(x.HiddenW, &r)
}
