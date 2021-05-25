using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeural_Network
{
    class Matrix_Math
    {
        private static Random r = new Random();
        private int rows, cols;
        public double[,] data;
        public Matrix_Math(double[,] data)
        {
            this.data = data;
            this.rows = data.GetLength(0);
            this.cols = data.GetLength(1);
        }
        public Matrix_Math(Matrix_Math m2)
        {
            // copy constractor

            this.rows = m2.data.GetLength(0);
            this.cols = m2.data.GetLength(1);
            this.data = new double[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i, j] = m2.data[i, j];
                }
            }
        }
        public Matrix_Math(int rows, int cols)
        {
            //build new matrix [rows*cols]
            this.rows = rows;
            this.cols = cols;
            data = new double[rows, cols];
        }
        public void Random_Weights()
        {
            //set Random values to matrics
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {
                    // recommended range 1/ number of weight into a node 
                    double num = 1 / Math.Sqrt(cols);
                    data[i, j] = RandomNumberBetween(-num, num);
                }
            }
        }
        public void randomlizeBias()
        {
            //set Random values to matrics between -1 to 1
            for (int i = 0; i < this.rows; i++)
            {
                for (int j = 0; j < this.cols; j++)
                {

                    double num = 1 / Math.Sqrt(cols);
                    data[i, j] = RandomNumberBetween(-1, 1);
                }
            }
        }
        private double RandomNumberBetween(double minValue, double maxValue)
        {
            double next = r.NextDouble();
            return minValue + (next * (maxValue - minValue));
        }
        public void PrintMatrix()
        {
            string s = "[";
            for (int i = 0; i < rows; i++)
            {
                for (int l = 0; l < cols; l++)
                {
                    if (data[i, l] >= 0)
                        s += ' ';
                    s += data[i, l];
                    if (l < cols - 1)
                        s += " , ";
                }
                if (i < rows - 1)
                    s += "\n ";
            }
            Console.WriteLine(s + "]\n");
        }
        public int GetRows() { return rows; }
        public int GetCols() { return cols; }
        public Matrix_Math multedbySigmoid()
        {
            // build new matrix with values of this matrix after sigmoid function
            Matrix_Math m2 = new Matrix_Math(this);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m2.data[i, j] = Math_Helper.sigmoid(this.data[i, j]);
                }
            }
            return m2;
        }
        public Matrix_Math dot(Matrix_Math m2)
        {
            // dot condition
            if (this.cols != m2.rows)
            {
                Console.WriteLine("Columns of A must match rows of B");
                return null;
            }
            Matrix_Math m3 = new Matrix_Math(GetRows(), m2.GetCols());
            // loop over m1 rows
            for (int i = 0; i < GetRows(); i++)
            {
                // loop over m2 cols
                for (int l = 0; l < m2.GetCols(); l++)
                {
                    double sum = 0;
                    // loop over each value in m2 cols 
                    for (int j = 0; j < GetCols(); j++)
                    {
                        sum += data[i, j] * m2.data[j, l];
                    }
                    m3.data[i, l] = sum;
                }
            }
            return m3;
        }
        public Matrix_Math Subtract(Matrix_Math m2)
        {
            // subtract this - m2 and retruns it 
            Matrix_Math m3 = new Matrix_Math(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m3.data[i, j] = this.data[i, j] - m2.data[i, j];
                }
            }
            return m3;
        }
        public Matrix_Math Add(Matrix_Math m2)
        {
            // add this + m2 and retruns it 
            Matrix_Math m3 = new Matrix_Math(rows, cols);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m3.data[i, j] = this.data[i, j] + m2.data[i, j];
                }
            }
            return m3;
        }
        public void AddSelf(Matrix_Math m2)
        {
            // add this + m2 

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i, j] += m2.data[i, j];
                }
            }

        }
        public Matrix_Math Transpose()
        {
            Matrix_Math m3 = new Matrix_Math(cols, rows);
            // loop over the rows of this matrix
            for (int i = 0; i < rows; i++)
            {
                // loop over the cols of this matrix
                for (int j = 0; j < cols; j++)
                {
                    // change the indexes
                    m3.data[j, i] = this.data[i, j];
                }
            }
            return m3;

        }
        public double GetValueFromMatrix(int row, int cols)
        {
            return data[row, cols];
        }
        public void SetValueToMatrix(int row, int cols, double item)
        {
            data[row, cols] = item;
        }
        public Matrix_Math UpdateFormula_dev()
        {
            Matrix_Math m2 = new Matrix_Math(this);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m2.data[i, j] = Math_Helper.dsigmoid(this.data[i, j]);
                }
            }
            return m2;
        }
        public void multSelf(double num)
        {
            // build new matrix with values of this matrix after sigmoid function

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    data[i, j] *= num;
                }
            }

        }
        public void SubtractSelf(Matrix_Math m2)
        {
            // subtract this - m2 
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    this.data[i, j] -= m2.data[i, j];
                }
            }

        }



    }
}
