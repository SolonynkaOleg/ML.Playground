using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Softvision.ML.NET.Playground.CreditScoring.DataStructures
{
    public class ObligorData
    {
        [LoadColumn(0)]
        public string Male { get; set; }
        [LoadColumn(1)]
        public float Age { get; set; }
        [LoadColumn(2)]
        public float Debt { get; set; }
        [LoadColumn(3)]
        public string Married { get; set; }
        [LoadColumn(4)]
        public string BankCustomer { get; set; }
        [LoadColumn(5)]
        public string EducationLevel { get; set; }
        [LoadColumn(6)]
        public string Ethnicity { get; set; }
        [LoadColumn(7)]
        public float YearsEmployed { get; set; }
        [LoadColumn(8)]
        public string PriorDefault { get; set; }
        [LoadColumn(9)]
        public string Employed { get; set; }
        [LoadColumn(10)]
        public float CreditScore { get; set; }
        [LoadColumn(11)]
        public string DriversLicense { get; set; }
        [LoadColumn(12)]
        public string Citizen { get; set; }
        [LoadColumn(13)]
        public string ZipCode { get; set; }
        [LoadColumn(14)]
        public float Income { get; set; }
        [LoadColumn(15)]
        public string Approved { get; set; }
    }
}
