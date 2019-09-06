using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Softvision.ML.NET.Playground.CreditScoring.DataStructures;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Softvision.ML.NET.Playground.CreditScoring
{
    class Program
    {
        static void Main(string[] args)
        {
            Run();
        }

        public static void Run()
        {
            MLContext mlContext = new MLContext();

            IDataView data = mlContext.Data.LoadFromTextFile<ObligorData>("crx.data.csv", separatorChar: ',');

            var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;

            var featureNames = new string[] {
                nameof(ObligorData.Male),
                nameof(ObligorData.Age),
                nameof(ObligorData.Debt),
                nameof(ObligorData.Married),
                nameof(ObligorData.BankCustomer),
                nameof(ObligorData.EducationLevel),
                nameof(ObligorData.Ethnicity),
                nameof(ObligorData.YearsEmployed),
                nameof(ObligorData.PriorDefault),
                nameof(ObligorData.Employed),
                nameof(ObligorData.CreditScore),
                nameof(ObligorData.DriversLicense),
                nameof(ObligorData.Citizen),
                nameof(ObligorData.ZipCode),
                nameof(ObligorData.Income)
            };

            var keyColumnName = nameof(ObligorData.Approved);

            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[] {
                    new InputOutputColumnPair(nameof(ObligorData.Male)),
                    new InputOutputColumnPair(nameof(ObligorData.Married)),
                    new InputOutputColumnPair(nameof(ObligorData.BankCustomer)),
                    new InputOutputColumnPair(nameof(ObligorData.EducationLevel)),
                    new InputOutputColumnPair(nameof(ObligorData.Ethnicity)),
                    new InputOutputColumnPair(nameof(ObligorData.PriorDefault)),
                    new InputOutputColumnPair(nameof(ObligorData.Employed)),
                    new InputOutputColumnPair(nameof(ObligorData.DriversLicense)),
                    new InputOutputColumnPair(nameof(ObligorData.Citizen)),
                    new InputOutputColumnPair(nameof(ObligorData.ZipCode))
                })
                .Append(mlContext.Transforms.ReplaceMissingValues(new[] {
                                          new InputOutputColumnPair(nameof(ObligorData.Age)),
                                          new InputOutputColumnPair(nameof(ObligorData.ZipCode))
                                      }))
                .Append(mlContext.Transforms.Concatenate("Features", featureNames))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression(labelColumnName: keyColumnName, featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, keyColumnName, "Score");

            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
            Console.ReadKey();
        }

        public static void PrintMultiClassClassificationMetrics(string name, CalibratedBinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.Accuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }
    }
}

