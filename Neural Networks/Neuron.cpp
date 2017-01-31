
#include "stdafx.h"
#include "NeuralNetworks.h"


using namespace NeuralNetworks;

	Neuron::Neuron(int inputCount) {
		_weights = new double[inputCount];
		_prevWeights = new double[inputCount];
		inputSize = inputCount;
	}

	double *Neuron::GetWeights() {
		return _weights;
	}

	double *Neuron::GetPrevWeights() {
		return _prevWeights;
	}

	void Neuron::SetCommonProperties(int inputCount){
		_weights = new double[inputCount];
		_prevWeights = new double[inputCount];
		inputSize = inputCount;
	}

	double Neuron::Response(double inputSignals[], int size)
	{
		// Check if the argument is correct.
		if (inputSignals == NULL || size != inputSize)
			throw "The signal array must have the same length as the weight array.";

		// Calculate the output as the sum of products.
		double result = 0.0;
		for (int i = 0; i < inputSize; i++)
			result += _weights[i] * inputSignals[i];
		return result;
	}

	double Neuron::Response(double inputSignals[], int size, double *odl) {
		if (inputSignals == NULL || size != inputSize)
			throw "The signal array must have the same length as the weight array.";

		double result = 0.0;
		*odl = 0;
		for (int i = 0; i < inputSize; i++)
			*odl += pow(_weights[i] - inputSignals[i], 2);

		result = 1 / (1E-10 + *odl);
		return result;
	}

	double Neuron::Strength(double signals[], int size, StrengthNorm norm)
	{
		double strength = 0;
		switch (norm)
		{
		case Manhattan:
			for (int i = 0; i < size; i++)
				strength += abs(signals[i]);
			return strength;

		case Euclidean:
			for (int i = 0; i < size; i++)
				strength += signals[i] * signals[i];
			return sqrt(strength);

		default:
			return 0.0;
		}
	}

	void Neuron::Normalize(double signals[], int size)
	{
		double strength = Strength(signals, size, Euclidean);
		for (int i = 0; i < size; i++)
			signals[i] /= strength;
	}

	double Neuron::MemoryTraceStrength(StrengthNorm norm)
	{
		return Strength(_weights, inputSize, norm);
	}

	void Neuron::Learn(double signals[], int size, double expectedOutput, double ratio,
		double *previousResponse, double *previousError)
	{
		*previousResponse = Response(signals, size);
		*previousError = expectedOutput - *previousResponse;
		for (int i = 0; i < inputSize; i++)
			_weights[i] += ratio * *previousError * signals[i];
	}

	void Neuron::Learn(double signals[], int size, double previous_weights[],
		double error, double sigma, double ratio, double momentum)
	{
		for (int i = 0; i < inputSize; i++)
			_weights[i] += ratio * sigma * signals[i] - momentum * (_weights[i] - previous_weights[i]);
	}

	void Neuron::Randomize(double min, double max)
	{
		double length = max - min;
		for (int i = 0; i < inputSize; i++)
			//for the random func we can use some other numbers
			_weights[i] = min + length * random();
	}

	void Neuron::Learn(double signals[], int size, double etha, double max)
	{

		double previous_response = Response(signals, size);

		if (previous_response < 0.2 * max)
			previous_response *= 0.3;
		if (previous_response < 0)
			previous_response *= 0.1;

		for (int i = 0; i < inputSize; i++)
		{
			_prevWeights[i] = _weights[i];
			_weights[i] += etha * previous_response * (signals[i] - _weights[i]);
		}
	}

	void Neuron::LearnSelf(double signals[], int size, double etha)
	{
		for (int i = 0; i < inputSize; i++)
		{
			_prevWeights[i] = _weights[i];
			_weights[i] += etha * (signals[i] - _weights[i]);
		}
	}

	void Neuron::Randomize(double min, double max, double epsilon)
	{
		double length = max - min;
		for (int i = 0; i < inputSize; i++)
		{
			_weights[i] = min + length * random();
			if (abs(_weights[i]) < epsilon)
				_weights[i] = epsilon;
		}
	}

	int  Neuron::GetINputSize() {
		return inputSize;
	}

	double Neuron::random(double min, double max)
	{
		return (double)(rand()) / RAND_MAX*(max - min) + min;
	}

	