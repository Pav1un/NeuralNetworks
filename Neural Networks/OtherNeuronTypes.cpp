
#include "stdafx.h"
#include "NeuralNetworks.h"


using namespace NeuralNetworks;

//----NonLinearNeuron---------
	double NonLinearNeuron::Response(double inputSignals[], int size)
	{
		return ActivationFunction(Neuron::Response(inputSignals, size));
	}

	void NonLinearNeuron::LearnWidrowHoff(double signals[], int size, double expectedOutput,
		double ratio, double *previousResponse, double *previousError)
	{
		// Take the linear response into consideration.
		*previousResponse = Neuron::Response(signals, size);
		*previousError = expectedOutput - *previousResponse;
		for (int i = 0; i < Neuron::GetINputSize(); i++)
			Neuron::GetWeights()[i] += ratio * *previousError * signals[i];
	}
//----NonLinearNeuron---------


//----BipolarNeuron---------
	double BipolarNeuron::ActivationFunction(double arg)
	{
		return arg > 0 ? 1 : -1;
	}
//----BipolarNeuron---------


//----UnipolarNeuron---------
    double UnipolarNeuron::ActivationFunction(double arg)
	{
		return arg > 0 ? 1 : 0;
	}
//----UnipolarNeuron---------


//----SigmoidalNeuron---------
	double SigmoidalNeuron::ActivationFunction(double arg)
	{
		return 1.0 / (1.0 + exp(-_beta * arg));
	}

	double SigmoidalNeuron::GetBeta() 
	{
		return _beta;
	}

	void SigmoidalNeuron::SetBeta(double value)
	{
		_beta = value;
	}
//----SigmoidalNeuron---------

	
//----TanhNeuron---------
	 double TanhNeuron::ActivationFunction(double arg)
	{
		return tanh(arg);
	}
//----TanhNeuron---------