using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Topology
    {
        public int InputCount { get; }
        public int OutputCount { get; }
        public List<int> HiddenLayers { get; }
        public double LeraningRate { get; }

        public Topology(int inputCount, int outputCount, double learninRate, params int[] layers)
        {
            InputCount = inputCount;
            OutputCount = outputCount;
            LeraningRate = learninRate;
            HiddenLayers = new List<int>();
            HiddenLayers.AddRange(layers);
        }
    }
}
