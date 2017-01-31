#include "stdafx.h"
#include "NeuralNetworks.h"


using namespace NeuralNetworks;

	TeachingSet::TeachingSet(int inputCount, int outputCount)
	{
		_inputCount = inputCount;
		_outputCount = outputCount;
	}

	TeachingSet::TeachingSet(TeachingSet set, int setSize) : list(setSize)
	{
		_inputCount = set._inputCount;
		_outputCount = set._outputCount;
		_teachSetSize = setSize;
		for (auto setItr = set.cbegin(); setItr != set.end(); setItr++) {
			push_back(*setItr);
		}

	}

	int TeachingSet::GetInputCount() {
		return _inputCount;
	}

	void TeachingSet::SetInputCount(int value) {
		_inputCount = value;
	}

	int TeachingSet::GetOutputCount() {
		return _outputCount;
	}

	void TeachingSet::SetOutputCount(int value) {
		_outputCount = value;
	}