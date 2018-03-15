/*
General Approach to Analyse ML Problem with small data
Stage(1) - Getting Raw data.
Stage(2) - Data Exploration
            1. Quantitative Analysis
                a. Variable Identification.
                b. Distribution Analysis on Continuous variables.
                    i. Outlier detection.
                    ii. Skewness detection.
                c. Null Detection on all variables.
            2. Feature Identification - i.e. identifying features that influence Target Variable.
                a. Categorical-Target variable graphical Analysis.
                b. Continuous-Target variable graphical Analysis.
                c. Hypothesis Testing.
                d. Correlation.
            3. Feature Munging
                a. Null Removal
                c. Outlier Removal
Stage(3) - Setting Pipeline Stages
            i. Feature Extractors
            ii. Problem Solver
Stage(4) - Dataset Splitting [(Training), (Validation) and (Testing)]
Stage(5) - Problem Solver Preparation.
            i. Setting up Pipeline/Estimator.
            ii. Setting up Parameter grid.
            iii. Setting up Evaluator.
Stage(6) - Hyperparameter Tuning using (Training) and (Validation).
Stage(7) - Best Model Testing using (Testing).
Stage(8) - Best Model Evaluation.
Stage(9) - Saving/Showing Results to Repository
            i. Testing Results.
            ii. Evaluation Results.
            iii. The best model.
Note:- To ensure expected results, print Schemas for Data set generated in between stages.
*/

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Row, SparkSession, functions}
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.feature.{Bucketizer, ChiSqSelector, IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}

object LoanAnalysis {
  def main(args: Array[String]): Unit = {
    // Comment this to view all Log4j messages
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(1) - Getting Raw data.
    val sqlContext : SparkSession = SparkSession.builder().appName("Loan Predictive Analysis").master("local[*]").getOrCreate()
    var trainingRawDF : DataFrame = sqlContext.read.format("csv").option("inferSchema","true").option("header","true").load(".\\LoanData\\LoanTrainData.csv")
    var testingRawDF : DataFrame = sqlContext.read.format("csv").option("inferSchema","true").option("header","true").load(".\\LoanData\\LoanTestData.csv")
    trainingRawDF.show(10, truncate = false)
    trainingRawDF.printSchema()

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(2) - Data Exploration
    // 1. Quantitative Analysis
    //    a. Variable Identification.
    //       Categorical Variables : Gender, Married, Dependents, Education, Self_Employed, Credit_History, Property_Area, Loan_Status
    //       Continuous Variables : ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term
    trainingRawDF.describe().show(false)
    //    b. Distribution Analysis on Continuous variables.
    //       i. Outlier detection : ApplicantIncome, LoanAmount, CoapplicantIncome
    //       ii. Skewness detection.
    //          CoapplicantIncome - right skewed + not normal distribution
    //          ApplicantIncome - slightly right skewed + normal distribution
    //          LoanAmount - slightly right skewed + not normal distribution
    //    c. Null Detection on all variables.
    for(column <- trainingRawDF.columns)
      println(f"Null values in column $column%s : ${trainingRawDF.filter( trainingRawDF(column) === null || trainingRawDF(column).isNull || trainingRawDF(column).isNaN ).count()}%d")
    // 2. Feature Identification - i.e. identifying features that influence Target Variable.
    //    a. Categorical-Target variable Graphical Analysis.
    //    b. Continuous-Target variable Graphical Analysis.
    //    c. Hypothesis Testing : Performed dynamically within Pipeline by ChiSqSelecter
    //    d. Correlation
    var corrTestDF : DataFrame = trainingRawDF.select("ApplicantIncome","CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")
    corrTestDF.stat.freqItems(corrTestDF.columns,0.5).show(false)
    // Filling column wise nulls with corresponding mode before Feature transformation else an exception is thrown
    corrTestDF = corrTestDF.na.fill(Map(("ApplicantIncome",4583),("CoapplicantIncome",240),("LoanAmount",133),("Loan_Amount_Term",360)))
    for(column <- corrTestDF.columns)
      corrTestDF = corrTestDF.withColumn(column, corrTestDF(column).cast(DoubleType))
    val vectorAssembler : VectorAssembler = new VectorAssembler().setInputCols(Array("ApplicantIncome","CoapplicantIncome", "LoanAmount", "Loan_Amount_Term")).setOutputCol("Feature")
    corrTestDF = vectorAssembler.transform(corrTestDF)
    corrTestDF.show(false)
    val corrResult : Row = Correlation.corr(corrTestDF,"Feature","pearson").head()
    println(f"Correlation in Continuous variables : \n${corrResult.toString()}%s")
    /*
    Conclusion :
    a. No correlation of "Loan_Amount_Term" with any of "ApplicantIncome","CoapplicantIncome", "LoanAmount" or it is completely independent.
    b. "CoapplicantIncome" variates slightly -ve to "ApplicantIncome" and slightly +ve to "LoanAmount".
    c. "ApplicantIncome" variates nicely +ve to "LoanAmount".
    Inference : "LoanAmount" is decided on basis of "ApplicantIncome" and "CoapplicantIncome".
    */

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // 3. Data Munging
    //    a. Null Removal
    for(column <- Array("Credit_History","ApplicantIncome","CoapplicantIncome","LoanAmount")) {
      trainingRawDF = trainingRawDF.withColumn(column, trainingRawDF(column).cast(DoubleType))
      testingRawDF = testingRawDF.withColumn(column, testingRawDF(column).cast(DoubleType))
    }
    trainingRawDF = trainingRawDF.na.fill(Map(("Gender","Male"),("Credit_History",1),("Dependents","2"),("Married","Yes"),("Education","Graduate"),("Self_Employed","No"),("Property_Area","Semiurban"),("ApplicantIncome",3812.50),("CoapplicantIncome",1239.5),("LoanAmount",126),("Loan_Amount_Term",360)))
    testingRawDF = testingRawDF.na.fill(Map(("Gender","Male"),("Credit_History",1),("Dependents","2"),("Married","Yes"),("Education","Graduate"),("Self_Employed","No"),("Property_Area","Semiurban"),("ApplicantIncome",3812.50),("CoapplicantIncome",1239.5),("LoanAmount",126),("Loan_Amount_Term",360)))
    println("Checking for Null removal : ")
    for(column <- trainingRawDF.columns)
      println(f"TrainingDF Null values in column $column%s : ${trainingRawDF.filter(trainingRawDF(column) === null || trainingRawDF(column).isNull || trainingRawDF(column).isNaN).count()}%d")
    //    b. Outlier Removal
    for (column <- Array("ApplicantIncome","CoapplicantIncome","LoanAmount")) {
      trainingRawDF = trainingRawDF.withColumn("cubeRoot".concat(column), functions.pow(trainingRawDF(column), 0.3333))
      testingRawDF = testingRawDF.withColumn("cubeRoot".concat(column), functions.pow(testingRawDF(column), 0.3333))
    }

    trainingRawDF = trainingRawDF.na.fill(Map(("cubeRootApplicantIncome",0.0),("cubeRootCoapplicantIncome",0.0),("cubeRootLoanAmount",0.0)))
    testingRawDF = testingRawDF.na.fill(Map(("cubeRootApplicantIncome",0.0),("cubeRootCoapplicantIncome",0.0),("cubeRootLoanAmount",0.0)))
    // Output of Stage(2)
    val trainingFilteredDF : DataFrame = trainingRawDF.select("Loan_ID","Loan_Status","Gender","Married","Dependents","Education","Credit_History","Self_Employed","Property_Area","Loan_Amount_Term","cubeRootApplicantIncome","cubeRootCoapplicantIncome","cubeRootLoanAmount")
    val testingFilteredDF : DataFrame = testingRawDF.select("Loan_ID","Gender","Married","Dependents","Education","Credit_History","Self_Employed","Property_Area","cubeRootApplicantIncome","Loan_Amount_Term","cubeRootCoapplicantIncome","cubeRootLoanAmount")
    trainingFilteredDF.printSchema()
    trainingFilteredDF.show(15, truncate = false)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(3) - Setting Pipeline Stages
    // i. Feature Extractors
    val genderToIndex : StringIndexer = new StringIndexer().setInputCol("Gender").setOutputCol("IndexedGender").setHandleInvalid("keep")
    val marriedToIndex : StringIndexer = new StringIndexer().setInputCol("Married").setOutputCol("IndexedMarried").setHandleInvalid("keep")
    val dependentsToIndex : StringIndexer = new StringIndexer().setInputCol("Dependents").setOutputCol("IndexedDependents").setHandleInvalid("keep")
    val educationToIndex : StringIndexer = new StringIndexer().setInputCol("Education").setOutputCol("IndexedEducation").setHandleInvalid("keep")
    val employedToIndex : StringIndexer = new StringIndexer().setInputCol("Self_Employed").setOutputCol("IndexedSelfEmployed").setHandleInvalid("keep")
    val propertyToIndex : StringIndexer = new StringIndexer().setInputCol("Property_Area").setOutputCol("IndexedPropertyArea").setHandleInvalid("keep")
    val termBucketizer : Bucketizer = new Bucketizer().setInputCol("Loan_Amount_Term").setOutputCol("BucketedTerm").setSplits(Array(Double.NegativeInfinity,150,200,350,450,Double.PositiveInfinity))
    val labelToIndex : StringIndexer = new StringIndexer().setInputCol("Loan_Status").setOutputCol("IndexedLoanStatus").setHandleInvalid("keep")
    val categoricalFeatureMaker : VectorAssembler = new VectorAssembler().setInputCols(Array("IndexedGender","IndexedMarried","IndexedDependents","IndexedEducation","IndexedSelfEmployed","IndexedPropertyArea","Credit_History","BucketedTerm")).setOutputCol("CategoricalFeature")
    val chiSqSelector : ChiSqSelector = new ChiSqSelector().setFeaturesCol("CategoricalFeature").setOutputCol("SelectedCategoricalFeature").setLabelCol("IndexedLoanStatus").setNumTopFeatures(7)
    val featureMaker : VectorAssembler = new VectorAssembler().setInputCols(Array("SelectedCategoricalFeature","cubeRootApplicantIncome","cubeRootCoapplicantIncome")).setOutputCol("Feature")
    val indexToLabel : IndexToString = new IndexToString().setInputCol("Prediction").setOutputCol("PredictedLoanStatus").setLabels(labelToIndex.fit(trainingRawDF).labels)
    // ii. Problem Solver
    val problemSolver : LogisticRegression = new LogisticRegression().setFeaturesCol("Feature").setLabelCol("IndexedLoanStatus").setPredictionCol("Prediction").setMaxIter(20)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(4) - Dataset Splitting [(Training) and (Validation)]
    val splitData : Array[DataFrame] = trainingFilteredDF.randomSplit(Array(0.8,0.2), seed = 11L)
    val (trainingDF : DataFrame, validationDF : DataFrame) = (splitData(0),splitData(1))

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(5) - Problem Solver Preparation.
    // i. Setting up Pipeline/Estimator.
    val pipeLine : Pipeline = new Pipeline().setStages(Array(genderToIndex,marriedToIndex,dependentsToIndex,educationToIndex,employedToIndex,propertyToIndex,termBucketizer,labelToIndex,categoricalFeatureMaker,chiSqSelector,featureMaker,problemSolver,indexToLabel))
    // ii. Setting up Parameter grid.
    val paramGrid : Array[ParamMap] = new ParamGridBuilder().addGrid(problemSolver.regParam, Array(0.1,0.01,0.001)).addGrid(problemSolver.elasticNetParam, Array(0.10,0.30,0.50)).build()

    // iii. Setting up Evaluator.
    val evaluator : BinaryClassificationEvaluator = new BinaryClassificationEvaluator().setLabelCol("IndexedLoanStatus").setMetricName("areaUnderROC")

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(6) - Hyperparameter Tuning using (Training) and (Validation).
    val crossValidator : CrossValidator = new CrossValidator().setEstimator(pipeLine).setEstimatorParamMaps(paramGrid).setEvaluator(evaluator).setNumFolds(5)
    val crossValidatorModel : CrossValidatorModel = crossValidator.fit(trainingDF)
    val crossValidatorDF : DataFrame = crossValidatorModel.transform(validationDF)
    crossValidatorDF.printSchema()
    crossValidatorDF.show(10, truncate = false)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(7) - Best Model Testing using (Testing).
    val bestPipelineModel : PipelineModel = crossValidatorModel.bestModel.asInstanceOf[PipelineModel]
    val testingDF : DataFrame = bestPipelineModel.transform(testingFilteredDF)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(8) - Best Model Evaluation.
    val areaUnderROC : Double = evaluator.evaluate(crossValidatorDF)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    // Stage(9) - Saving/Showing Results to Repository
    // i. Evaluation Results.
    println(f"Area under ROC over Validation Set : ${areaUnderROC*100}%2.2f")
    // ii. Testing Results.
    testingDF.show(false)
    testingDF.select("Loan_ID","PredictedLoanStatus").withColumnRenamed("PredictedLoanStatus","Loan_Status").write.format("csv").option("inferSchema","true").option("header","true").save(".\\LoanData\\sample_submission")
    // iii. The best model.
    val bestLogisticModel : LogisticRegressionModel =  bestPipelineModel.stages(11).asInstanceOf[LogisticRegressionModel]
    println(f"Best Model Parameters\n1.RegParam : ${bestLogisticModel.getRegParam}%f")
    println(f"2.Tolerance : ${bestLogisticModel.getElasticNetParam}%f")
  }
}
