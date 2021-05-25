using System;

namespace BasicNeural_Network
{
    class Program
    {
        static void Main(string[] args)
        {




            Random r = new Random();
            NeuralNetwork n = new NeuralNetwork(0.1, 2, new int[] { 05, 05 }, 1);

            for (int i = 0; i < 1500; i++)
            {
                int l = r.Next(0, 4);
                if (l == 0)
                {
                    n.train(new double[,] { { 0 }, { 0 } }, new double[,] { { 0 } });
                }
                if (l == 1)
                {
                    n.train(new double[,] { { 1 }, { 0 } }, new double[,] { { 1 } });
                }
                if (l == 2)
                {
                    n.train(new double[,] { { 0 }, { 1 } }, new double[,] { { 1 } });
                }
                if (l == 3)
                {
                    n.train(new double[,] { { 1 }, { 1 } }, new double[,] { { 0 } });
                }
            }



            n.feedForward(new double[,] { { 0 }, { 0 } }).PrintMatrix();
            n.feedForward(new double[,] { { 1 }, { 1 } }).PrintMatrix();

            n.feedForward(new double[,] { { 1 }, { 0 } }).PrintMatrix();
            n.feedForward(new double[,] { { 0 }, { 1 } }).PrintMatrix();


        }



    }
}
