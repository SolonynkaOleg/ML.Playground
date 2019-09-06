using Microsoft.ML.Data;

namespace Softvision.ML.NET.Playground.CreditScoring.Simplified
{
    public class ObligorData
    {
        [LoadColumn(9)]
        public string Employed { get; set; }
        [LoadColumn(14)]
        public float Income { get; set; }
        [LoadColumn(10)]
        public float CreditScore { get; set; }
    }
}
