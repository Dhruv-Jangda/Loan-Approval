import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{DataFrame, SparkSession}

object HypothesisTester {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val sqlContext : SparkSession = SparkSession.builder().appName("Hypothesis Tester").master("local[*]").getOrCreate()
    val trainingRawDF : DataFrame = sqlContext.read.format("csv").option("inferSchema","true").option("header","true").load(".\\LoanData\\LoanTrainData.csv")

    // i. H0 - Credit_History independent of Loan_Status. Intuition : Those having Credit_History as 1.0 very probable to get Loan.
    var hypothesisDF : DataFrame = trainingRawDF.groupBy("Loan_Status").pivot("Credit_History").count()
    var array_H0 : Array[Double] = hypothesisDF.select("0","1").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    var matrix_H0 = Matrices.dense(2,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 1 : Credit_History independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
    // Conclusion : H0 rejected OR "Credit_History" to be selected while making model.

    // ii. H0 - "Dependents" independent of "Loan_Status".
    hypothesisDF  = trainingRawDF.groupBy("Loan_Status").pivot("Dependents").count()
    array_H0 = hypothesisDF.select("0","1","2","3+").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    matrix_H0 = Matrices.dense(4,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 2 : Dependents independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
    // Conclusion : H0 rejected OR "Dependents" to be selected while making model.

    // iii. H0 - "Education" independent of "Loan_Status". Intuition : Those having "Education" as Graduate more probable than Not-Graduate.
    hypothesisDF  = trainingRawDF.groupBy("Loan_Status").pivot("Education").count()
    array_H0 = hypothesisDF.select("Graduate","Not Graduate").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    matrix_H0 = Matrices.dense(2,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 3 : Type of Education independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
    // Conclusion : H0 rejected OR "Education" to be selected while making model.

    // iv. H0 - "Gender" independent of "Loan_Status".
    hypothesisDF  = trainingRawDF.groupBy("Loan_Status").pivot("Gender").count()
    array_H0 = hypothesisDF.select("Female","Male").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    matrix_H0 = Matrices.dense(2,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 4 : Type of Gender independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
    // Conclusion : H0 accepted OR "Gender" to be rejected while making model.

    // v. H0 - "Married" independent of "Loan_Status". Intuition : Those "Married" are more probable than not married.
    hypothesisDF  = trainingRawDF.groupBy("Loan_Status").pivot("Married").count()
    array_H0 = hypothesisDF.select("No","Yes").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    matrix_H0 = Matrices.dense(2,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 5 : Married Status independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
    // Conclusion : H0 rejected OR "Married" to be selected while making model

    // vi. H0 - "Property_Area" independent of "Loan_Status". Intuition : Those having "Property_Area" as Semi-Urban more probable than Rural and Urban.
    hypothesisDF  = trainingRawDF.groupBy("Loan_Status").pivot("Property_Area").count()
    array_H0 = hypothesisDF.select("Rural","Semiurban","Urban").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    matrix_H0 = Matrices.dense(3,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 6 : Property_Area independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
    // Conclusion : H0 rejected OR "Property_Area" to be selected while making model.

    // vii - "Self_Employed" independent of "Loan_Status".
    hypothesisDF  = trainingRawDF.groupBy("Loan_Status").pivot("Self_Employed").count()
    array_H0 = hypothesisDF.select("No","Yes").collect().flatMap(row => row.mkString(",").split(",").map(_.toDouble))
    matrix_H0 = Matrices.dense(2,2,array_H0).transpose
    hypothesisDF.show(false)
    println(matrix_H0.toString())
    println("Null Hypothesis 7 : Self_Employed independent of Loan_Status")
    println(Statistics.chiSqTest(matrix_H0).toString())
  }
}
