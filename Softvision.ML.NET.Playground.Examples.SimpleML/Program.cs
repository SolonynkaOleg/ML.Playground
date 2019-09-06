using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace Softvision.ML.NET.Playground.Examples.SimpleML
{
    class Program
    {
        static string[] featureNames = new string[] {
                nameof(CreditData.Male),
                nameof(CreditData.Age),
                nameof(CreditData.Debt),
                nameof(CreditData.Married),
                nameof(CreditData.BankCustomer),
                nameof(CreditData.EducationLevel),
                nameof(CreditData.Ethnicity),
                nameof(CreditData.YearsEmployed),
                nameof(CreditData.PriorDefault),
                nameof(CreditData.Employed),
                nameof(CreditData.CreditScore),
                nameof(CreditData.DriversLicense),
                nameof(CreditData.Citizen),
                nameof(CreditData.ZipCode),
                nameof(CreditData.Income)
            };
        static void Main(string[] args)
        {
            // Step 1. Create ML Context
            MLContext mlContext = new MLContext();

            // Step 2. Read data from file
            IDataView data = mlContext.Data.LoadFromTextFile<CreditData>("crx.data.csv", separatorChar: ',');

            // Step 3. Split data into train and test set
            var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            // Step 4. Build data process pipeline
            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[] {
                    new InputOutputColumnPair(nameof(CreditData.Male)),
                    new InputOutputColumnPair(nameof(CreditData.Married)),
                    new InputOutputColumnPair(nameof(CreditData.BankCustomer)),
                    new InputOutputColumnPair(nameof(CreditData.EducationLevel)),
                    new InputOutputColumnPair(nameof(CreditData.Ethnicity)),
                    new InputOutputColumnPair(nameof(CreditData.PriorDefault)),
                    new InputOutputColumnPair(nameof(CreditData.Employed)),
                    new InputOutputColumnPair(nameof(CreditData.DriversLicense)),
                    new InputOutputColumnPair(nameof(CreditData.Citizen)),
                    new InputOutputColumnPair(nameof(CreditData.ZipCode))
                })
                .Append(mlContext.Transforms.Concatenate("Features", featureNames));

            // Step 5. Append trainer to your pipeline
            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: nameof(CreditData.Approved), featureColumnName: "Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            // Step 6. Train your model
            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            // Step 7. Evaluate your model against test data
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, nameof(CreditData.Approved), "Score");

            PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
            Console.ReadKey();
        }

        public static void PrintBinaryClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*       Metrics for {name} binary classification model      ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");
        }
    }
}
