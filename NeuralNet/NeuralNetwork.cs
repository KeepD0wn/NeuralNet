using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class NeuralNetwork
    {
        public List<Layer> Layers { get; }
        public Topology Topology { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayers();
            CreateHiddenLayrs();
            CreateOutputLayer();
        }


        /// <summary>
        /// создаём выводящий слой и добавляем нейрон
        /// </summary>
        private void CreateOutputLayer()
        {
            var outputNeurons = new List<Neuron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

        /// <summary>
        /// создаём обрабатывающий слой и добавляем туда нейроны
        /// </summary>
        private void CreateHiddenLayrs()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeurons = new List<Neuron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++) //ну так
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var hiddenLayer = new Layer(hiddenNeurons);
                Layers.Add(hiddenLayer);
            }
        }

        /// <summary>
        /// создаём принимающий слой и добавляем туда нейроны
        /// </summary>
        private void CreateInputLayers()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }

        /// <summary>
        /// задаём сигналы(выход знач) всем нейронам и возвращаем выводимый нейрон
        /// </summary>
        /// <param name="inputSignals">сигналы для входа</param>
        /// <returns></returns>
        public Neuron Predict(params double[] inputSignals)
        {
            SendSignalsToInputNeutrons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(n => n.Output).First();
            }
        }

        /// <summary>
        /// прогоняет датасет чрез нейронку столько сколько эпох
        /// </summary>
        /// <param name="dataset"></param>
        /// <param name="epoch"></param>
        /// <returns></returns>
        public double Learn(double[] expected, double[,] inputs, int epoch)
        {
            var signalsNormal = Normalization(inputs);
            var signals = Scalling(signalsNormal);

            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += BackPropagation(output, input);
                }
            }

            var result = error / epoch;
            return result;
        }

        public static double[] GetRow(double[,] matrix, int row)
        {
            var collumns = matrix.GetLength(1);
            var array = new double[collumns];
            for (int i = 0; i < collumns; i++)
                array[i] = matrix[row, i];
            return array;
        }

        private double[,] Scalling(double[,] inputs) 
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int collumn = 0; collumn < inputs.GetLength(1); collumn++)
            {
                var min = inputs[0, collumn];
                var max = inputs[0, collumn];

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, collumn];

                    if (item < min)
                    {
                        min = item;
                    }

                    if (item > max)
                    {
                        max = item;
                    }
                }

                var divider = max - min;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, collumn] = (inputs[row, collumn] - min) / divider;
                    if (Double.IsNaN(result[row, collumn]))
                        result[row, collumn] = 0;
                }
            }

            return result;
        }

        private double[,] Normalization(double[,] inputs)
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int collumn = 0; collumn < inputs.GetLength(1); collumn++)
            {
                //среднее значение сигнала нейрона
                var sum = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, collumn];
                }
                var average = sum / inputs.GetLength(0);

                //стандартное квадратичное отклонение нейрона
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, collumn] - average), 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, collumn] = (inputs[row, collumn] - average) / standartError;
                    if (Double.IsNaN(result[row, collumn]))
                        result[row, collumn] = 0;
                }
            }
            return result;
        }

        /// <summary>
        /// метод обратного распространения ошибки
        /// </summary>
        /// <returns></returns>
        private double BackPropagation(double expected, params double[] inputs)
        {
            var actual = Predict(inputs).Output;

            var difference = actual - expected;

            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LeraningRate);
            }

            for (int i = Layers.Count - 2; i >= 0; i--)
            {
                var layer = Layers[i];
                var previousLayer = Layers[i + 1];

                for (int j = 0; j < layer.NeuronCount; j++)
                {
                    var neuron = layer.Neurons[j];

                    for (int k = 0; k < previousLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previousLayer.Neurons[k];
                        var error = previousNeuron.Weights[j] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LeraningRate);
                    }
                }
            }

            var result = difference * difference;
            return result;
        }

        /// <summary>
        /// задаёт выходное значения для кажого нейрона в своём слое
        /// </summary>
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }

        /// <summary>
        /// задаём значения входа
        /// </summary>
        /// <param name="inputSignals"></param>
        public void SendSignalsToInputNeutrons(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];

                neuron.FeedForward(signal);
            }
        }
    }
}
