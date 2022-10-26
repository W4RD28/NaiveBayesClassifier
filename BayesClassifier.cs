using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NaiveBayes
{
    class BayesClassifier
    {
        private Dictionary<string, int>[] stringToInt; // "male" -> 0, etc.
        private int[][][] jointCounts; // [feature][value][dependent]
        private int[] dependentCounts;

        public BayesClassifier()
        {
            this.stringToInt = null; // need training data to know size
            this.jointCounts = null; //  need training data to know size
            this.dependentCounts = null; //  need training data to know size
        }

        public void Train(string[][] trainData)
        {
            // 1. scan training data and construct one dictionary per column
            int numRows = trainData.Length;
            int numCols = trainData[0].Length;
            this.stringToInt = new Dictionary<string, int>[numCols]; // allocate array

            for (int col = 0; col < numCols; ++col) // including y-values
            {
                stringToInt[col] = new Dictionary<string, int>(); // instantiate Dictionary

                int idx = 0;

                for (int row = 0; row < numRows; ++row) // each row of curr column
                {
                    string s = trainData[row][col];
                    if (stringToInt[col].ContainsKey(s) == false) // first time seen
                    {
                        stringToInt[col].Add(s, idx); // ex: analyst -> 0
                        ++idx;
                    }
                } // each row

            } // each col

            // 2. scan and count using stringToInt Dictionary
            this.jointCounts = new int[numCols - 1][][]; // do not include the y-value

            // a. allocate second dim
            for (int c = 0; c < numCols - 1; ++c) // each feature column but not y-column
            {
                int count = this.stringToInt[c].Count; // number possible values for column
                jointCounts[c] = new int[count][];
            }

            // b. allocate last dimension = always 2 for binary classification
            for (int i = 0; i < jointCounts.Length; ++i)
                for (int j = 0; j < jointCounts[i].Length; ++j)
                {
                    //int numDependent = stringToInt[stringToInt.Length - 1].Count;
                    //jointCounts[i][j] = new int[numDependent];
                    jointCounts[i][j] = new int[2]; // binary classification
                }

            // c. init joint counts with 1 for Laplacian smoothing
            for (int i = 0; i < jointCounts.Length; ++i)
                for (int j = 0; j < jointCounts[i].Length; ++j)
                    for (int k = 0; k < jointCounts[i][j].Length; ++k)
                        jointCounts[i][j][k] = 1;

            // d. compute joint counts
            for (int i = 0; i < numRows; ++i)
            {
                string yString = trainData[i][numCols - 1]; // dependent value
                int depIndex = stringToInt[numCols - 1][yString]; // corresponding index
                for (int j = 0; j < numCols - 1; ++j)
                {
                    int attIndex = j;

                    string xString = trainData[i][j]; // an attribute value like "male"

                    int valIndex = stringToInt[j][xString]; // corresponding integer like 0

                    ++jointCounts[attIndex][valIndex][depIndex];
                }
            }

            // 3. scan and count number of each of the 2 dependent values
            this.dependentCounts = new int[2]; // binary
            for (int i = 0; i < dependentCounts.Length; ++i) // Laplacian init
                dependentCounts[i] = numCols - 1; // numCols - 1 = num features

            for (int i = 0; i < trainData.Length; ++i)
            {
                string yString = trainData[i][numCols - 1]; // conservative or liberal
                int yIndex = stringToInt[numCols - 1][yString]; // 0 or 1
                ++dependentCounts[yIndex];
            }

            return;  // the trained 'model' is jointCounts and dependentCounts
        } // Train

        public double Probability(string yValue, string[] xValues)
        {
            int numFeatures = xValues.Length; // ex: 3 (job, sex, income)
            double[][] conditionals = new double[2][]; // binary

            for (int i = 0; i < 2; ++i)
                conditionals[i] = new double[numFeatures]; // ex: P('doctor' | conservative)

            double[] unconditionals = new double[2]; // ex: P('conservative'), P('liberal')

            // convert strings to ints
            int y = this.stringToInt[numFeatures][yValue];
            int[] x = new int[numFeatures];

            for (int i = 0; i < numFeatures; ++i)
            {
                string s = xValues[i];
                x[i] = this.stringToInt[i][s];
            }

            // compute conditionals
            for (int k = 0; k < 2; ++k) // each y-value
            {
                for (int i = 0; i < numFeatures; ++i)
                {
                    int attIndex = i;
                    int valIndex = x[i];
                    int depIndex = k;

                    conditionals[k][i] =
                        (jointCounts[attIndex][valIndex][depIndex] * 1.0) / dependentCounts[depIndex];
                }
            }

            // compute unconditionals
            int totalDependent = 0; // ex: count(conservative) + count(liberal)
            for (int k = 0; k < 2; ++k)
                totalDependent += this.dependentCounts[k];

            for (int k = 0; k < 2; ++k)
                unconditionals[k] = (dependentCounts[k] * 1.0) / totalDependent;

            // compute partials
            double[] partials = new double[2];
            for (int k = 0; k < 2; ++k)
            {
                partials[k] = 1.0; // because we are multiplying
                for (int i = 0; i < numFeatures; ++i)
                    partials[k] *= conditionals[k][i];
                partials[k] *= unconditionals[k];
            }

            // evidence = sum of partials
            double evidence = 0.0;
            for (int k = 0; k < 2; ++k)
                evidence += partials[k];

            return partials[y] / evidence;
        } // Probability

        public double Accuracy(string[][] data)
        {
            int numCorrect = 0;
            int numWrong = 0;
            int numRows = data.Length;
            int numCols = data[0].Length;

            for (int i = 0; i < numRows; ++i) // row
            {
                string yValue = data[i][numCols - 1]; // assumes y in last column
                string[] xValues = new string[numCols - 1];
                Array.Copy(data[i], xValues, numCols - 1);

                double p = this.Probability(yValue, xValues);

                if (p > 0.50)
                    ++numCorrect;
                else
                    ++numWrong;
            }

            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }
    }
}
