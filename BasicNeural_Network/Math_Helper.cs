using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeural_Network
{
    static class Math_Helper
    {
        public static double sigmoid(double x)
        {
            return (Math.Pow(Math.E, x)) / (Math.Pow(Math.E, x) + 1);
        }
        public static double dsigmoid(double x)
        {
            // sigmoid(x) * (1-sigmoid(x))
            return x * (1 - x); 
        }
        public static Matrix_Math MultiplyHadamard(Matrix_Math m1, Matrix_Math m2)
        {
            // multply all the values in m1 and m2
            Matrix_Math Result = new Matrix_Math(m1.GetRows(), m1.GetCols());
            for (int i = 0; i < m1.GetRows(); i++)
            {
                for (int l = 0; l < m1.GetCols(); l++)
                {
                    Result.data[i, l] = m1.data[i, l] * m2.data[i, l];
                }
            }
            return Result;
        }


    }
}
