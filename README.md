# Loan Approval

A machine learning pipeline with Logistic Regression as a problem solver to predict if a loan will be approved for the applicants based on given data. The input training data is split to training and validation set, where pipeline is fitted with training and evaluated on validation. The best Pipeline Model is used to make predictions over the test data with output as a csv file.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

1. IntelliJ IDEA with Scala plugin.
2. Spark 2.2.0.
3. Tableau.

### Development Environment Setup

1. Download IntelliJ IDEA Community edition from https://www.jetbrains.com/idea/download/#section=windows.
2. Install it.
3. Download and embed Scala plugin to IntelliJ IDEA :
	1. While running it the first time, a window with recommended plugins is shown.
	2. In case if the window is skipped, follow : File -> Settings -> Plugins in Settings window. In search box, type Scala and install it.
4. Download spark-2.2.0-bin-hadoop2.7.tgz from https://spark.apache.org/downloads.html.
5. Extract it to a location of your choice say **location_1** and name it to your choice say **name**.
6. Download the project Loan Approval to **location_2**.
7. Open IntelliJ IDEA.
8. Go To Open and browse to **location_2**.
9. Select the project Loan Approval and click open.
10. Go To File -> Project Structure.
11. In Project Structure window, select Libraries from left panel.
12. Click on + -> Java.
13. In new window popped, go to **location_1** -> **name** -> jars and click OK.
14. Name the library imported to your choice and click on OK.
15. Make sure to verify the library imported by checking for the green tick at upper right corner of code editor.
16. Now click Build -> Build Project.
17. Rectify errors if any.
18. Now go to src in left pane of code editor and right click on LoanAnalysis, then on Run 'LoanAnalysis'.
19. Go to **location_2** using windows explorer and check csv file at LoanData -> sample_submission.

Note :
1. Spark 2.2.0 is compatible with Scala version 2.11.11.
2. Tableau is needed for data exploration.

## Running the tests

1. Check dataframe schemas after a stage to ensure if the stage is working as expected.
2. Check csv file produced by pipeline model at **location_2** -> LoanData -> sample_submission.

## Live Deployment

In order to deploy the project to live clustered environment :
1. Create jar file of the project using IntelliJ IDEA as :
	1. Open project Loan Approval in IntelliJ IDEA.
	2. Go to File -> Project Structure.
	3. In Project Structure window, select Artifacts from left panel.
	4. Click on + -> JAR and select option as needed.
	5. Specify settings as needed and click OK.
	6. Now, a jar file will be created when the project is build.
2. Use the created jar file in spark-submit command in bin directory of **name**(folder name of extracted spark-2.2.0-bin-hadoop2.7.tgz) and refer https://spark.apache.org/docs/latest/submitting-applications.html for further settings and use of spark-submit.

## Author

**Dhruv Jangda** https://github.com/Dhruv-Jangda