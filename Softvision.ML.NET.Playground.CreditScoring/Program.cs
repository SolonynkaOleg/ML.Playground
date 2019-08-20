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
            MLContext mlContext = new MLContext();

            IDataView data = mlContext.Data.LoadFromTextFile<ObligorData>("cdx.data.csv", separatorChar: ',');

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

            var dataProcessPipeline = mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.Male))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.Married)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.BankCustomer)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.EducationLevel)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.Ethnicity)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.PriorDefault)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.Employed)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.DriversLicense)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.Citizen)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.ZipCode)))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(nameof(ObligorData.Approved)))
                .Append(mlContext.Transforms.Concatenate("Features", featureNames))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(ObligorData.Approved)))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "KeyColumn", featureColumnName: "Features");

            var trainingPipeline = dataProcessPipeline.Append(trainer);

            ITransformer trainedModel = trainingPipeline.Fit(trainData);

            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");

            PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
            Console.ReadKey();
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            Console.WriteLine($"************************************************************");
            Console.WriteLine($"*    Metrics for {name} multi-class classification model   ");
            Console.WriteLine($"*-----------------------------------------------------------");
            Console.WriteLine($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            Console.WriteLine($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            Console.WriteLine($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            Console.WriteLine($"************************************************************");
        }
    }
}
