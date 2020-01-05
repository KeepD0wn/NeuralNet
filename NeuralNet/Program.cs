using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Program
    {
        static void Main(string[] args)
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();
            using (var sr = new StreamReader("Health.csv"))
            {
               // var header = sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    var values = row.Split(';').Select(v => Convert.ToDouble(v)).ToList();
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                    inputs.Add(input);
                }
            }

            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for (int j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }

            var topology = new Topology(outputs.Count, 1, 0.05, outputs.Count / 2);
            var neuralNetwork = new NeuralNetwork(topology);
            var difference = neuralNetwork.Learn(outputs.ToArray(), inputSignals, 30000);

            var results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                var res = neuralNetwork.Predict(inputs[i]).Output;
                results.Add(res);
            }


            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 3);
                var actual = Math.Round(results[i], 3);
                Console.WriteLine("ожидание " + expected);
                Console.WriteLine("реальность " + actual);
                Console.WriteLine();
            }

            var res2 = neuralNetwork.Predict(1, 1, 1, 1).Output;
            Console.WriteLine("рез(1) " + Math.Round(res2, 3));

            Console.ReadKey();
        }
    }
}
