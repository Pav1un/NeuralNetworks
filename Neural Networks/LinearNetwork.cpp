#include "stdafx.h"
#include "NeuralNetworks.h"

using namespace NeuralNetworks;

	LinearNetwork::LinearNetwork(int numberOfInputs, int numberOfNeurons)
	{
		_neurons = new Neuron[numberOfNeurons];
		_neuronsCount = numberOfNeurons;
		_inputCount = numberOfInputs;
		for (int i = 0; i < numberOfNeurons; i++)
			_neurons[i].SetCommonProperties(numberOfInputs);
	}

	/*
	LinearNetwork(double[, ] initialWeights)
	: base(initialWeights.GetLength(1), initialWeights.GetLength(0))
	{
	for (int neuron = 0; neuron < _neurons.Length; neuron++)
	for (int input = 0; input < _inputCount; input++)
	_neurons[neuron].Weights[input] = initialWeights[neuron, input];
	}*/

	Neuron *LinearNetwork::GetNeurons() {
		return _neurons;
	};

	double *LinearNetwork::GetNeurons_dist() {
		return _neurons_dist;
	};

	int LinearNetwork::GetInputCount() {
		return _inputCount;
	}

	double *LinearNetwork::Response(double inputSignals[], int size)
	{
		if (inputSignals == NULL || size != _inputCount)
			throw "The signal array's length must be equal to the number of inputs.";

		double *res = new double[_neuronsCount];
		for (int i = 0; i < _neuronsCount; i++)
			res[i] = _neurons[i].Response(inputSignals, size);

		return res;
	}

	double *LinearNetwork::Response(double inputSignals[], int size, int *num)
	{
		if (inputSignals == NULL || size != _inputCount)
			throw "The signal array's length must be equal to the number of inputs.";

		double *res = new double[_neuronsCount];
		double max = 0;
		double dist = 0;

		*num = -1;

		_neurons_dist = new double[_neuronsCount];

		for (int i = 0; i < _neuronsCount; i++)

		{
			res[i] = _neurons[i].Response(inputSignals, size, &dist);

			_neurons_dist[i] = dist;
			if (res[i] > max)
			{
				max = res[i];
				*num = i;
			}
		}

		return res;
	}

	int LinearNetwork::Winner(double output[], int size, double threshold)
	{
		int result = -1;
		double max = threshold;
		for (int i = 0; i < size; i++)
			if (output[i] > max)
			{
				max = output[i];
				result = i;
			}
		return result;
	}

	double LinearNetwork::Winner(double inputSignals[], int size)
	{
		double max = 0;

		double res;
		for (int i = 0; i < _neuronsCount; i++)
		{
			res = _neurons[i].Response(inputSignals, size);
			if (abs(res) > max)
				max = res;
		}
		return max;
	}

	class LearningStatistics
	{
	public: double *Output;
			double Error;
			LearningStatistics(double *output, double error)
			{
				Output = output;
				Error = error;
			};
	};

	void LinearNetwork::Learn(Element teachingElement, double etha)
	{
		double _max = 0;


		_max = Winner(teachingElement.Inputs, teachingElement.inputsCount);

		for (int x = 1; x < _neuronsCount; x++)
		{

		}

		for (int neuronIndex = 0; neuronIndex < _neuronsCount; neuronIndex++)
		{
			_neurons[neuronIndex].Learn(
				teachingElement.Inputs,
				teachingElement.inputsCount,
				etha,
				_max
			);
		}
	}

	void LinearNetwork::Learn(Element teachingElement, int neuronIndex, double etha)
	{
		_neurons[neuronIndex].LearnSelf(
			teachingElement.Inputs,
			teachingElement.inputsCount,
			etha
		);
	}

	void LinearNetwork::Learn(Element teachingElement, double ratio,
		double *previousResponse, double *previousError)
	{
		if (previousResponse == NULL)
			previousResponse = new double[_neuronsCount];
		if (previousError == NULL)
			previousError = new double[_neuronsCount];

		for (int neuronIndex = 0; neuronIndex < _neuronsCount; neuronIndex++)
		{
			_neurons[neuronIndex].Learn(
				teachingElement.Inputs,
				teachingElement.inputsCount,
				teachingElement.ExpectedOutputs[neuronIndex],
				ratio,
				&previousResponse[neuronIndex],
				&previousError[neuronIndex]
			);
		}
	}

	void LinearNetwork::Randomize(double min, double max)
	{
		for (int i = 0; i < _neuronsCount; i++)
			_neurons[i].Randomize(min, max);
	}

	void LinearNetwork::Randomize(double min, double max, double epsilon)
	{
		for (int i = 0; i < _neuronsCount; i++)
			_neurons[i].Randomize(min, max, epsilon);
	};

	