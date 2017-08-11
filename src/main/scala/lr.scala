package com.test

import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util
import java.util.{Calendar, Date}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.sql.DataFrame

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.根据onehot编码，输入到lr进行计算
  */
object lr {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  //  val seed_file = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  val partition_num = 1000
  val test = true
  //tree
  var numIterations = 50
  // 100
  var maxDepth = 3
  //6
  val learningRate = 0.5
  val minInstancesPerNode = 10
  var feature_size = 0

  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    println("***********************start*****************************")
    val sparkConf: SparkConf = new SparkConf().setAppName("xueyuan_lookalike")
    sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)
    println("***********************sc*****************************")
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    hiveContext = new HiveContext(sc)
    println("***********************hive*****************************")
    //    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    val uxip_boot_users_cycle = 7

    val table_in = "algo.lookalike_feature_onehot"
    var imeiFile = ""
    //目标用户数
    var targetSize = 0
    //0不包含种子用户，1包含种子用户
    var inclusive = 0
    var outputFile = ""
    println("***********************args_size=" + args.length + "*****************************")
    println("***********************args=" + args.mkString(",") + "*****************************")
    if (args.length >= 4) {
      imeiFile = args(0)
      println("***********************imeiFile=" + imeiFile + "*****************************")
      targetSize = args(1).toInt
      println("***********************targetSize=" + targetSize + "*****************************")
      inclusive = args(2).toInt
      println("***********************inclusive=" + inclusive + "*****************************")
      outputFile = args(3)
      println("***********************outputFile=" + outputFile + "*****************************")
      if (args.length >= 6) {
        numIterations = args(4).toInt
        maxDepth = args(5).toInt
        println("***********************numIterations=" + numIterations + "*****************************")
        println("***********************maxDepth=" + maxDepth + "*****************************")
      }
    }
    //seed
    val seed = get_seed(imeiFile)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed.length + "*****************************")

    //all user = seed + other
    val user_feature = load_data_onehot(table_in)
    user_feature.cache()
    //    user_feature.repartition(partition_num)
    //seed
    val seeduser_feature = user_feature.filter(r => seed.contains(r._1.toString)).mapPartitions(iter => for (r <- iter) yield (r._1, r._2))
    seeduser_feature.cache()
    val seeduser_feature_size = seeduser_feature.count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seeduser_feature_size=" + seeduser_feature_size + "*****************************")

    //other
    val user_feature_cyc = user_feature.filter(r => (r._3.contains("," + uxip_boot_users_cycle + ","))).mapPartitions(iter => for (r <- iter) yield (r._1, r._2))
    val otheruser_feature = user_feature_cyc.filter(r => (seed.contains(r._1.toString) == false))
    user_feature_cyc.cache()
    otheruser_feature.cache()
    val otheruser_feature_size = otheruser_feature.count()
    user_feature.unpersist()

    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************otheruser_feature_size=" + otheruser_feature_size + "*****************************")

    //sample
    var samp = seeduser_feature_size.toDouble * 10 / otheruser_feature_size
    if (samp > 1) {
      samp = 1
    }
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************samp=" + samp + "*****************************")
    val sampuser_feature = otheruser_feature.sample(false, samp, 1L)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************sampuser_feature_size=" + sampuser_feature.count() + "*****************************")
    //positive
    val point_seed = seeduser_feature.mapPartitions(iter => for (r <- iter) yield {
      new LabeledPoint(1.0, r._2)
    })
    //negative
    val point_samp = sampuser_feature.mapPartitions(iter => for (r <- iter) yield {
      new LabeledPoint(0.0, r._2)
    })
    //training
    val model = training_lr(point_seed, point_samp)
    seeduser_feature.unpersist()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************model finished *****************************")
    //predict
    var user_feature_forpre: RDD[(Long, SparseVector)] = null
    if (inclusive == 1) {
      user_feature_forpre = user_feature_cyc
      otheruser_feature.unpersist()
    } else {
      user_feature_forpre = otheruser_feature
      user_feature_cyc.unpersist()
    }
    val result = pre(model, user_feature_forpre, targetSize)
    if (inclusive == 1) {
      user_feature_cyc.unpersist()
    } else {
      otheruser_feature.unpersist()
    }
    save(outputFile, result)


  }

  def save(outputFile: String, result: Array[(Long, Double)]): Unit = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val outputstream = hdfs.create(new Path(outputFile), true)
    for ((imei, score) <- result) {
      outputstream.writeBytes(imei + "," + score + "\n")
    }
    outputstream.close()
    if (test) {
      val outputstream2 = hdfs.create(new Path(outputFile + "_1w"), true)
      for ((imei, score) <- result) {
        outputstream2.writeBytes(imei + "\n")
      }
      outputstream2.close()
    }
  }

  def get_seed(imeiFile: String): Array[String] = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val path1 = new Path(imeiFile)
    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
    var seed_id = new ArrayBuffer[String]()
    var line1 = reader1.readLine()
    while (line1 != null) {
      if (!line1.equals("null")) {
        seed_id += line1.trim
      }
      line1 = reader1.readLine()
    }
    seed_id.toArray
  }


  def training_reg(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint]) = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************numIterations=" + numIterations + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************learningRate=" + learningRate + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************maxDepth=" + maxDepth + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************minInstancesPerNode=" + minInstancesPerNode + "*****************************")
    val data = point_seed ++ point_samp
    data.repartition(partition_num)
    var boostingStrategy = BoostingStrategy.defaultParams("Regression") //Classification  Regression
    boostingStrategy.setNumIterations(numIterations) //Note: Use more iterations in practice.
    boostingStrategy.setLearningRate(learningRate)
    boostingStrategy.treeStrategy.setMaxDepth(maxDepth)
    //    boostingStrategy.treeStrategy.setMinInstancesPerNode(minInstancesPerNode)
    val categoricalFeaturesInfo = new util.HashMap[Integer, Integer]()
    for (i <- 0 until feature_size) {
      categoricalFeaturesInfo.put(i, 2)
    }
    boostingStrategy.treeStrategy.setCategoricalFeaturesInfo(categoricalFeaturesInfo)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training start*****************************")
    val model = GradientBoostedTrees.train(data, boostingStrategy)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training end*****************************")
    //    val scoreAndLabels = data.map { point =>
    //      val prediction = model.predict(point.features)
    //      (prediction, point.label)
    //    }
    val scoreAndLabels = data.mapPartitions(iter => for (point <- iter) yield {
      val prediction = model.predict(point.features)
      (prediction, point.label)
    })
    scoreAndLabels.cache()
    for ((p, l) <- scoreAndLabels.take(100)) {
      println(p + "," + l)
    }
    val TP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 1)).count()
    val FP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 0)).count()
    val FN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 1)).count()
    val TN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 0)).count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************TP = " + TP + ", FP = " + FP + ", FN = " + FN + ", TN = " + TN + "*****************************")
    if (TP + FP != 0) {
      val p = TP.toDouble / (TP + FP)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************p = " + p + "*****************************")
    }
    if (TP + FN != 0) {
      val r = TP.toDouble / (TP + FN)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************r = " + r + "*****************************")
    }
    //    for ((label, pre) <- labelsAndPredictions_seed.take(10)) {
    //      println("*****************************(" + label + "," + pre + ")**********************************")
    //    }
    //    val testMSE_seed = scoreAndLabels.map { case (v, p) => math.pow((v - p), 2) }.mean()
    //    println("Test Mean Squared Error seed = " + testMSE_seed)
    //ev
    val metrics1 = new BinaryClassificationMetrics(scoreAndLabels)
    val metrics2 = new BinaryClassificationMetrics(scoreAndLabels.mapPartitions(iter => for (r <- iter) yield {
      if (r._1 > 0.5) {
        (1.0, r._2)
      } else {
        (0.0, r._2)
      }
    }))
    val auc1 = metrics1.areaUnderROC()
    val auc2 = metrics2.areaUnderROC()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************auc = " + auc1 + "," + auc2 + "*****************************")

    scoreAndLabels.unpersist()
    //    println("Learned regression GBT model:\n" + model.toDebugString)
    model

  }

  def training_lr(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint]) = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************numIterations=" + numIterations + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************learningRate=" + learningRate + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************maxDepth=" + maxDepth + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************minInstancesPerNode=" + minInstancesPerNode + "*****************************")
    val data = point_seed ++ point_samp
    data.repartition(partition_num)
    val lr = new LogisticRegressionWithLBFGS().setIntercept(true) //.setNumClasses(2)
    lr.optimizer.setRegParam(0.6).setNumIterations(5000)//setConvergenceTol(0.000001).
      .setUpdater(new SquaredL2Updater)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training start*****************************")
    val model = lr.run(data)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training end*****************************")
    //    val scoreAndLabels = data.map { point =>
    //      val prediction = model.predict(point.features)
    //      (prediction, point.label)
    //    }
    val scoreAndLabels = data.mapPartitions(iter => for (point <- iter) yield {
      val prediction = model.predict(point.features)
      (prediction, point.label)
    })
    scoreAndLabels.cache()
    for ((p, l) <- scoreAndLabels.take(100)) {
      println(p + "," + l)
    }
    val TP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 1)).count()
    val FP = scoreAndLabels.filter(r => (r._1 >= 0.5 && r._2 == 0)).count()
    val FN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 1)).count()
    val TN = scoreAndLabels.filter(r => (r._1 < 0.5 && r._2 == 0)).count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************TP = " + TP + ", FP = " + FP + ", FN = " + FN + ", TN = " + TN + "*****************************")
    if (TP + FP != 0) {
      val p = TP.toDouble / (TP + FP)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************p = " + p + "*****************************")
    }
    if (TP + FN != 0) {
      val r = TP.toDouble / (TP + FN)
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************r = " + r + "*****************************")
    }
    //    for ((label, pre) <- labelsAndPredictions_seed.take(10)) {
    //      println("*****************************(" + label + "," + pre + ")**********************************")
    //    }
    //    val testMSE_seed = scoreAndLabels.map { case (v, p) => math.pow((v - p), 2) }.mean()
    //    println("Test Mean Squared Error seed = " + testMSE_seed)
    //ev
    val metrics1 = new BinaryClassificationMetrics(scoreAndLabels)
    val metrics2 = new BinaryClassificationMetrics(scoreAndLabels.mapPartitions(iter => for (r <- iter) yield {
      if (r._1 > 0.5) {
        (1.0, r._2)
      } else {
        (0.0, r._2)
      }
    }))
    val auc1 = metrics1.areaUnderROC()
    val auc2 = metrics2.areaUnderROC()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************auc = " + auc1 + "," + auc2 + "*****************************")

    scoreAndLabels.unpersist()
    //    println("Learned regression GBT model:\n" + model.toDebugString)
    model

  }

  def pre(model: LogisticRegressionModel, user_feature_forpre: RDD[(Long, SparseVector)], targetSize: Int): Array[(Long, Double)] = {
    val user_pre = user_feature_forpre.repartition(partition_num).mapPartitions(iter => for (r <- iter) yield {
      val prediction = model.predict(r._2)
      (r._1, prediction)
    })
    val array = user_pre.collect().sortWith(_._2 > _._2)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************lookalike imei size = " + array.filter(_._2 > 0.5).length + "*****************************")
    if (targetSize > 0) {
      array.take(targetSize)
    } else {
      array
    }
  }

  def load_data_onehot(table_in: String): RDD[(Long, SparseVector, String)] = {
    var df: DataFrame = null
    val c1 = Calendar.getInstance()
    while (df == null || df.count() == 0) {
      c1.add(Calendar.DATE, -1)
      val sql_1 = "select imei,feature_size,feature,uxip_boot_users_cycle from " + table_in + " where app_cur_com_meizu_media_reader in (70,72,74,75,76,78,79) and stat_date=" + sdf_date.format(c1.getTime())
      df = hiveContext.sql(sql_1).cache()
    }

    val feature_size1 = df.take(1).map(r => r.getInt(1))
    feature_size = feature_size1(0)
    println("***********************load_data finished*****************************")
    df.repartition(partition_num)
    val feature_rdd = df.mapPartitions(iter => for (r <- iter) yield {
      val imei = r.getLong(0)
      val feature_size = r.getInt(1)
      val feature = r.getString(2)
      val cyc = r.getString(3)
      var feature_index = Array.empty[Int]
      if (feature != null && feature.length > 0) {
        feature_index = feature.split(",").map(index => index.toInt)
      }
      val sp = new SparseVector(feature_size, feature_index, Array.fill(feature_index.length)(1.0))
      (imei, sp, cyc)
    })
    feature_rdd
  }

  def save_data(table_out: String, user_pre: RDD[(Long, Double)]): Unit = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save table start*****************************")
    val candidate_rdd = user_pre.mapPartitions(iter => for (r <- iter) yield {
      Row(r._1, r._2)
    })

    val structType = StructType(
      StructField("imei", LongType, false) ::
        StructField("score", DoubleType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( imei BIGINT,score DOUBLE ) partitioned by (stat_date bigint) stored as textfile"
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val sdf1 = new SimpleDateFormat("yyyyMMdd")
    val date1 = sdf1.format(c1.getTime())
    val insertInto_table_sql: String = "insert overwrite table " + table_out + " partition(stat_date = " + date1 + ") select * from "
    val table_temp = "table_temp"
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save data start*****************************")
    candidate_df.registerTempTable(table_temp)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************register TempTable finished*****************************")
    hiveContext.sql(create_table_sql)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************create table finished*****************************")
    hiveContext.sql(insertInto_table_sql + table_temp)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************insertInto table finished*****************************")
  }


}
