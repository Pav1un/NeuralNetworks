// Приведенный ниже блок ifdef - это стандартный метод создания макросов, упрощающий процедуру 
// экспорта из библиотек DLL. Все файлы данной DLL скомпилированы с использованием символа NEURALNETWORKS_EXPORTS,
// в командной строке. Этот символ не должен быть определен в каком-либо проекте
// использующем данную DLL. Благодаря этому любой другой проект, чьи исходные файлы включают данный файл, видит 
// функции NEURALNETWORKS_API как импортированные из DLL, тогда как данная DLL видит символы,
// определяемые данным макросом, как экспортированные.
#ifdef NEURALNETWORKS_EXPORTS
#define NEURALNETWORKS_API __declspec(dllexport)
#else
#define NEURALNETWORKS_API __declspec(dllimport)
#endif

using namespace std;
namespace NeuralNetworks {
	class Neuron
	{

	public:
		Neuron() {};
		Neuron(int inputCount);
		enum StrengthNorm {
			Manhattan,
			Euclidean
		};
		double *GetWeights();
		double * GetPrevWeights() ;
		int GetINputSize();
		void SetCommonProperties(int inputCount);
		virtual double Response(double inputSignals[], int size);
		virtual double Response(double inputSignals[], int size, double *odl);
		static double Strength(double signals[], int size, StrengthNorm norm);
		static void Normalize(double signals[], int size);
		double MemoryTraceStrength(StrengthNorm norm);
		void Learn(double signals[], int size, double expectedOutput, double ratio,
			double *previousResponse, double *previousError);
		void Learn(double signals[], int size, double previous_weights[],
			double error, double sigma, double ratio, double momentum);
		void Randomize(double min, double max);
		void Learn(double signals[], int size, double etha, double max);
		void LearnSelf(double signals[], int size, double etha);
		void Randomize(double min, double max, double epsilon);

		double * PrevWeights;

	private:
		double *_weights;
		double *_prevWeights;
		int inputSize;
		
		double random(double min = 1.0, double max = 500.0);

	};

	class Element
	{
	public:
		double* Inputs;
		double *ExpectedOutputs;
		string Comment;
		int inputsCount;

		Element *Clone()
		{
			Element *ret = new Element;
			ret->Inputs = Inputs;
			ret->ExpectedOutputs = ExpectedOutputs;
			return ret;
		};
	};

	class LinearNetwork
	{
	private:
		Neuron *_neurons;
		double *_neurons_dist;
		int _inputCount;
		int _neuronsCount;

	public:
		LinearNetwork(int numberOfInputs, int numberOfNeurons);
		Neuron *GetNeurons();
		double *GetNeurons_dist();
		int GetInputCount();
		double *Response(double inputSignals[], int size);
		double *Response(double inputSignals[], int size, int *num);
		static int Winner(double output[], int size, double threshold);
		double Winner(double inputSignals[], int size);
		void Learn(Element teachingElement, double etha);
		void Learn(Element teachingElement, int neuronIndex, double etha);
		void Learn(Element teachingElement, double ratio,
			double *previousResponse, double *previousError);
		void Randomize(double min, double max);
		void Randomize(double min, double max, double epsilon);
		
	};

	

	class TeachingSet : list<Element>
	{

	private:
		int _inputCount;
		int _outputCount;
		int _teachSetSize;

	public: 
		TeachingSet(int inputCount, int outputCount);
		TeachingSet(TeachingSet set, int setSize);
		int GetInputCount();
		void SetInputCount(int value);
		int GetOutputCount();
		void SetOutputCount(int value);

	

		//next code should be refactored
		/*
		static double *ParseLine(TextReader reader, int expectedLength,
				string exceptionString)
		{		
			NumberFormatInfo invariantFormat =
				CultureInfo.InvariantCulture.NumberFormat;

			String line = reader.ReadLine();
			if (line == null)
				return null;
			if (line == "")
				throw "Empty line encountered where the numbers were expected.";

			string[] fields = line.Split(',');
			if (fields.Length != expectedLength)
				throw new ApplicationException(exceptionString);
			double[] ret = new double[expectedLength];
			for (int i = 0; i < fields.Length; i++)
				ret[i] = double.Parse(fields[i], invariantFormat);
			return ret;
		}

	
		static TeachingSet FromFile(string fileName)
		{
			using (StreamReader sr = new StreamReader(fileName))
			{
				return TeachingSet.FromText(sr);
			}
		}

	
		static TeachingSet FromStream(Stream stream)
		{
			using (StreamReader sr = new StreamReader(stream))
			{
				return TeachingSet.FromText(sr);
			}
		}

	
		static TeachingSet FromText(TextReader reader)
		{
			try
			{
				string line = reader.ReadLine();
				if (line == NULL)
					return null;

				string[] fields = line.Split(',');
				if (fields.Length != 2)
					throw new ApplicationException("First line must contain two numbers.");
				int inputCount = int.Parse(fields[0]);
				int outputCount = int.Parse(fields[1]);

				TeachingSet ret = new TeachingSet(inputCount, outputCount);

				while ((line = reader.ReadLine()) != NULL)
				{
					Element elem;
					elem.Comment = line;

					elem.Inputs =
						ParseLine(reader, inputCount, "Invalid number of inputs.");
					if (elem.Inputs == null)
						break;

					elem.ExpectedOutputs =
						ParseLine(reader, outputCount, "Invalid number of outputs.");
					if (elem.ExpectedOutputs == null)
						break;

					ret.Add(elem);
				}
				return ret;
			}
			catch (ApplicationException) { throw; }
			catch (Exception ex)
			{
				throw new ApplicationException("The file is corrupted.", ex);
			}
		}

		void Normalize()
		{
			foreach(Element elem in this)
				Neuron.Normalize(elem.Inputs);
		}*/
	};


	class NonLinearNeuron : Neuron
	{
	public: 
		NonLinearNeuron(int inputCount) : Neuron(inputCount) {};

		virtual double ActivationFunction(double arg) = 0;
		virtual double Response(double inputSignals[], int size);
		void LearnWidrowHoff(double signals[], int size, double expectedOutput,
			double ratio, double *previousResponse, double *previousError);		

	};

	class BipolarNeuron : NonLinearNeuron
	{
	public:
		BipolarNeuron(int inputCount) : NonLinearNeuron(inputCount) {};
		virtual double ActivationFunction(double arg);

	};

	class UnipolarNeuron : NonLinearNeuron
	{
	public:
		UnipolarNeuron(int inputCount) : NonLinearNeuron(inputCount) { };
		virtual double ActivationFunction(double arg);

	};

	class SigmoidalNeuron : NonLinearNeuron
	{
	public:
		SigmoidalNeuron(int inputCount) : NonLinearNeuron(inputCount) {};	
		double GetBeta();
		void SetBeta(double value);
		virtual double ActivationFunction(double arg);

	private:
		double _beta = 1.0;

	};

	class TanhNeuron : NonLinearNeuron
	{
	public:
		TanhNeuron(int inputCount) : NonLinearNeuron(inputCount) { };
		virtual double ActivationFunction(double arg);
		
	};

}

