package network

import "math"

func sigmoidMat(i, j int, v float64) float64 {
	return sigmoid(v)
}

func sigmoidDerivativeMat(i, j int, v float64) float64 {
	return sigmoidDerivative(v)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func squareError(desired, actual float64) float64 {
	return 0.5 * math.Pow(desired-actual, 2)
}
