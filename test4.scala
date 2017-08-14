package com.mainProject

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.ml.feature.{HashingTF, IndexToString, NGram, RegexTokenizer, StopWordsRemover, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.tartarus.snowball.ext.PorterStemmer
/**
  * Created by ashu on 4/16/2017.
  */
object test4 {
  def main(args: Array[String]): Unit= {
    System.setProperty("hadoop.home.dir", "C:\\hadoop\\");
    val spark = SparkSession
      .builder
      .master("local")
      .appName("Ashwini")
      .config("spark.sql.warehouse.dir", "file:///C:\\Users\\ashu\\IdeaProjects\\Testing\\spark-warehouse")
      .getOrCreate()


    //val core_5 = spark.read.json("C:\\Users\\ashu\\Desktop\\AmznRevw\\Musical_Instruments_5.json")

      val core_5 = spark.read.json("file:///home/ubuntu/reviews_Electronics_5.json")

      import spark.implicits._

   // core_5.groupBy("overall").count().orderBy("overall").show()

    core_5.createOrReplaceTempView("reviews")

    var reviewsDF = spark.sql(
      """
  SELECT text, label, rowNumber FROM (
    SELECT
       reviews.overall AS label
      ,reviews.reviewText AS text
      ,row_number() OVER (PARTITION BY overall ORDER BY rand()) AS rowNumber
    FROM reviews
  ) reviews
  WHERE rowNumber <= 200000
  """
    )

    //reviewsDF.groupBy("label").count().orderBy("label").show()

    val training = reviewsDF.filter(reviewsDF("rowNumber") > 15000).select("text","label")
    val test = reviewsDF.filter(reviewsDF("rowNumber") <= 15000).select("text","label")



    val numTraining = training.count()
    val numTest = test.count()

    println(s"numTraining = $numTraining, numTest = $numTest")

    val regexTokenizer = { new RegexTokenizer()
      .setPattern("[a-zA-Z']+")
      .setGaps(false)
      .setInputCol("text")
    }

    val remover = { new StopWordsRemover()
      .setInputCol(regexTokenizer.getOutputCol)
    }


    val ngram2 = { new NGram()
      .setInputCol(remover.getOutputCol)
      .setN(2)
    }

    val ngram3 = { new NGram()
      .setInputCol(remover.getOutputCol)
      .setN(3)
    }

    val removerHashingTF = { new HashingTF()
      .setInputCol(remover.getOutputCol)
    }
    val ngram2HashingTF = { new HashingTF()
      .setInputCol(ngram2.getOutputCol)
    }
    val ngram3HashingTF = { new HashingTF()
      .setInputCol(ngram3.getOutputCol)
    }

    val assembler = { new VectorAssembler()
      .setInputCols(Array(removerHashingTF.getOutputCol, ngram2HashingTF.getOutputCol, ngram3HashingTF.getOutputCol))
    }

    val labelIndexer = { new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(reviewsDF)
    }
    val labelConverter = { new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    }

    val nb = { new NaiveBayes()
      .setLabelCol(labelIndexer.getOutputCol)
      .setFeaturesCol(assembler.getOutputCol)
      .setPredictionCol("prediction")
      .setModelType("multinomial")
    }

    val pipeline = { new Pipeline()
      .setStages(Array(regexTokenizer, remover, ngram2, ngram3, removerHashingTF, ngram2HashingTF, ngram3HashingTF, assembler, labelIndexer, nb, labelConverter))
    }

    val paramGrid = { new ParamGridBuilder()
      .addGrid(removerHashingTF.numFeatures, Array(1000,10000))
      .addGrid(ngram2HashingTF.numFeatures, Array(1000,10000))
      .addGrid(ngram3HashingTF.numFeatures, Array(1000,10000))
      .build()
    }
    val cv = { new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)
    }

    val cvModel = cv.fit(training)

    val predictions = cvModel.transform(test)


    val evaluator = { new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    }
    val precision = evaluator.evaluate(predictions)
    println("Test Error = " + (1.0 - precision))

    predictions.printSchema()



    predictions.select("label","prediction").repartition(1).write.csv("file:///home/ubuntu/TrainingData/final9")
      predictions.select("text").repartition(1).write.csv("file:///home/ubuntu/TrainingData/final10")


  }
}
