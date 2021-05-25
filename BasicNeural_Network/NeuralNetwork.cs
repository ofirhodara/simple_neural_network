using System;
using System.Collections.Generic;
using System.Text;

namespace BasicNeural_Network
{
    class NeuralNetwork
    {

        private int NumOfinput, NumOfoutput;
        private int[] NumOfhidden;// At least 1
        private double lr;
        private Matrix_Math Winput_hidden, WLasthidden_output, biasFinal;
        private Matrix_Math[] Whidden_hidden, bias_hidden;

        public NeuralNetwork(double lr, int input, int[] NumOfhidden, int output)
        {
            // inilialize the network properly - Suitble for 1 layer and more
            this.lr = lr;
            this.NumOfinput = input;
            this.NumOfoutput = output;
            this.NumOfhidden = NumOfhidden;

            // Set randomly weights between input-->hidden[0]
            this.Winput_hidden = new Matrix_Math(NumOfhidden[0], NumOfinput);
            Winput_hidden.Random_Weights();

            // Set randomly weights between hidden_hidden
            Whidden_hidden = new Matrix_Math[NumOfhidden.Length - 1];
            for (int i = 0; i < Whidden_hidden.Length; i++)
            {
                Whidden_hidden[i] = new Matrix_Math(NumOfhidden[i + 1], NumOfhidden[i]);
                Whidden_hidden[i].Random_Weights();
            }
            // set bias to all the hidden layers
            bias_hidden = new Matrix_Math[NumOfhidden.Length];
            for (int i = 0; i < bias_hidden.Length; i++)
            {
                bias_hidden[i] = new Matrix_Math(NumOfhidden[i], 1);
                bias_hidden[i].randomlizeBias();
            }
            // Set randomly weights between last_hidden-->input
            this.WLasthidden_output = new Matrix_Math(NumOfoutput, NumOfhidden[NumOfhidden.Length - 1]);
            WLasthidden_output.Random_Weights();
            this.biasFinal = new Matrix_Math(NumOfoutput, 1);
            biasFinal.randomlizeBias();

        }
        public Matrix_Math train(double[,] inputs_array, double[,] targets_array)
        {

            Matrix_Math inputs = new Matrix_Math(inputs_array);
            Matrix_Math targets = new Matrix_Math(targets_array);

            // feeding forward the neural network
            Matrix_Math[] hiddens_outputs = new Matrix_Math[NumOfhidden.Length];
            // dot between weight ans input from previous layer
            hiddens_outputs[0] = this.Winput_hidden.dot(inputs);
            // add the bias to each nueron in the next layer
            hiddens_outputs[0].AddSelf(bias_hidden[0]);
            // activation function - sigmoid
            hiddens_outputs[0].multedbySigmoid();
            // do the same to each layer in the hidden layers
            for (int i = 1; i < hiddens_outputs.Length; i++)
            {
                hiddens_outputs[i] = this.Whidden_hidden[i - 1].dot(hiddens_outputs[i - 1]);
                hiddens_outputs[i].AddSelf(bias_hidden[i]);
                hiddens_outputs[i].multedbySigmoid();
            }
            // calculate the final outputs
            Matrix_Math outputLayer_Inputs = this.WLasthidden_output.dot(hiddens_outputs[hiddens_outputs.Length - 1]);
            outputLayer_Inputs.AddSelf(biasFinal);
            Matrix_Math outputLayer_Outputs = outputLayer_Inputs.multedbySigmoid();



            // Start to Train the Network BY back propagation
            // Calculates the error for the last layer - (target - actual)
            Matrix_Math output_errors = targets.Subtract(outputLayer_Outputs);

            // updating the weights an biases between the last hidden to final layer
            Matrix_Math deltasLast = Math_Helper.MultiplyHadamard(outputLayer_Outputs.UpdateFormula_dev(), output_errors);
            deltasLast.multSelf(-this.lr);
            biasFinal.Subtract(deltasLast);
            deltasLast = deltasLast.dot(hiddens_outputs[hiddens_outputs.Length - 1].Transpose());
            WLasthidden_output.SubtractSelf(deltasLast);

            // calculates the error to each of the hidden layers 
            // array of hidden's layers errors
            Matrix_Math[] error_hiddens = new Matrix_Math[NumOfhidden.Length];
            // calculate the error of the last hidden layer 
            error_hiddens[error_hiddens.Length - 1] = WLasthidden_output.Transpose().dot(output_errors);
            // calculate the error of all the hidden layers
            for (int i = error_hiddens.Length - 2; i >= 0; i--)
            {
                // calculate the erros by dot the weights of each layer and the next layer errors
                error_hiddens[i] = Whidden_hidden[i].Transpose().dot(error_hiddens[i + 1]);
            }

            // update the weights and biases in the hidden layers!
            Matrix_Math[] deltas_hiddens = new Matrix_Math[Whidden_hidden.Length];
            for (int i = 0; i < Whidden_hidden.Length; i++)
            {
                // mult by element the (next_layer_output)*(1-the next_layer_output)*(the next_layer_errors)
                deltas_hiddens[i] = Math_Helper.MultiplyHadamard(hiddens_outputs[i + 1].UpdateFormula_dev(), error_hiddens[i + 1]);
                deltas_hiddens[i].multSelf(-this.lr);
                // update the biases
                bias_hidden[i + 1].AddSelf(deltas_hiddens[i]);
                // dot by the output of the previous layer
                deltas_hiddens[i] = deltas_hiddens[i].dot(hiddens_outputs[i].Transpose());
                // update the weights by: new=old-lr*(D_costFunction/D_Weight(i,i+1))
                Whidden_hidden[i].SubtractSelf(deltas_hiddens[i]);
            }

            // update the weight input_firstHidden and the bias of the first hidden layer
            Matrix_Math deltasFirst = Math_Helper.MultiplyHadamard(hiddens_outputs[0].UpdateFormula_dev(), error_hiddens[0]);
            deltasFirst.multSelf(-this.lr);
            bias_hidden[0].Subtract(deltasFirst);
            deltasFirst = deltasFirst.dot(inputs.Transpose());
            Winput_hidden.SubtractSelf(deltasFirst);

            return outputLayer_Outputs;
        }
        public Matrix_Math feedForward(double[,] inputs_array)
        {
            // feeding forward the neural network - Suitble for 1 layer and more

            Matrix_Math inputs = new Matrix_Math(inputs_array);
            Matrix_Math[] hiddens_outputs = new Matrix_Math[NumOfhidden.Length];
            // dot between weight ans input from previous layer
            hiddens_outputs[0] = this.Winput_hidden.dot(inputs);
            // add the bias to each nueron in the next layer
            hiddens_outputs[0].AddSelf(bias_hidden[0]);
            // activation function - sigmoid
            hiddens_outputs[0].multedbySigmoid();
            // do the same to each layer in the hidden layers
            for (int i = 1; i < hiddens_outputs.Length; i++)
            {
                hiddens_outputs[i] = this.Whidden_hidden[i - 1].dot(hiddens_outputs[i - 1]);
                hiddens_outputs[i].AddSelf(bias_hidden[i]);
                hiddens_outputs[i].multedbySigmoid();
            }
            // calculate the final outputs
            Matrix_Math outputLayer_Outputs = this.WLasthidden_output.dot(hiddens_outputs[hiddens_outputs.Length - 1]);
            outputLayer_Outputs.AddSelf(biasFinal);
            return outputLayer_Outputs.multedbySigmoid();
        }

        public Matrix_Math GetWinput_hidden()
        {
            return this.Winput_hidden;
        }
        public Matrix_Math GetBiasFinal()
        {
            return this.biasFinal;
        }
        public Matrix_Math[] GetBias_hidden()
        {
            return bias_hidden;
        }
        public Matrix_Math[] GetWhidden_hidden()
        {
            return Whidden_hidden;
        }
        public Matrix_Math GetWLasthidden_output()
        {
            return this.WLasthidden_output;
        }


     


    }
}
