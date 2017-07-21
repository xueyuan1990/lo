package com.test

import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
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
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.
  */
object gbdt_bk {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  //  val seed_file = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  val partition_num = 200


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
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

    val uxip_boot_users_cycle = 7
    //    val samp = 0.02
    val numIterations = 100
    val maxDepth = 7
    val learningRate = 0.5
    val table_in = "algo.lookalike_feature_onehot"
    var imeiFile = ""
    //目标用户数
    var targetSize = 0
    //0不包含种子用户，1包含种子用户
    var inclusive = 0
    var outputFile = ""
    println("***********************args_size=" + args.length + "*****************************")
    println("***********************args=" + args.mkString(",") + "*****************************")
    if (args.length == 4) {
      imeiFile = args(0)
      println("***********************imeiFile=" + imeiFile + "*****************************")
      targetSize = args(1).toInt
      println("***********************targetSize=" + targetSize + "*****************************")
      inclusive = args(2).toInt
      println("***********************inclusive=" + inclusive + "*****************************")
      outputFile = args(3)
      println("***********************outputFile=" + outputFile + "*****************************")
    }
    //seed
    val seed = get_seed(imeiFile)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed.length + "*****************************")

    //all user = seed + other
    val user_feature = load_data_onehot(table_in, uxip_boot_users_cycle)
    user_feature.cache()
    user_feature.repartition(partition_num)
    //seed
    val seeduser_feature = user_feature.filter(r => seed.contains(r._1.toString)).mapPartitions(iter => for (r <- iter) yield (r._1, r._2))
    seeduser_feature.cache()
    val seeduser_feature_size = seeduser_feature.count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seeduser_feature_size=" + seeduser_feature_size + "*****************************")
    val point_seed = get_lable_feature_rdd(seeduser_feature, 1)
    //other
    val otheruser_feature = user_feature.filter(r => (seed.contains(r._1.toString) == false && r._3.contains("," + uxip_boot_users_cycle + ","))).mapPartitions(iter => for (r <- iter) yield (r._1, r._2))
    otheruser_feature.cache()
    val otheruser_feature_size = otheruser_feature.count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************otheruser_feature_size=" + otheruser_feature_size + "*****************************")

    //sample
    var samp = seeduser_feature_size.toDouble * 10 / otheruser_feature_size
    if (samp > 1) {
      samp = 1
    }
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************samp=" + samp + "*****************************")
    val sampuser_feature = otheruser_feature.sample(false, samp)
    val point_samp = get_lable_feature_rdd(sampuser_feature, 0)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************sampuser_feature_size=" + sampuser_feature.count() + "*****************************")

    //training
    val model = training_reg(point_seed, point_samp, numIterations, maxDepth, learningRate)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************model finished *****************************")
    //    predict

    var user_forpre = otheruser_feature;
    if (inclusive == 1) {
      user_forpre = user_feature.filter(r => (r._3.contains("," + uxip_boot_users_cycle + ","))).mapPartitions(iter => for (r <- iter) yield (r._1, r._2))
    }
    val user_feature_forpre = user_forpre.mapPartitions(iter => for (r <- iter) yield (r._1, new DenseVector(r._2)))
    val result = pre(model, user_feature_forpre, targetSize)
    //    save_data(outputFile, sc.parallelize(result))
    user_feature.unpersist()
    save(outputFile, result)


  }

  def save(outputFile: String, result: Array[(Long, Double)]): Unit = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val outputstream = hdfs.create(new Path(outputFile), true)
    for ((imei, score) <- result) {
      outputstream.writeBytes(imei + "," + score + "\n")
    }
    outputstream.close()
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


  def get_lable_feature_rdd(user_feature: RDD[(Long, Array[Double])], label: Int): RDD[LabeledPoint] = {
    val lable_feature_rdd = user_feature.mapPartitions(iter => for (r <- iter) yield {
      val dv = new DenseVector(r._2)
      (label, dv)
    })
    lable_feature_rdd.mapPartitions(iter => for (r <- iter) yield {
      new LabeledPoint(r._1, r._2)
    })
  }

  def training_reg(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint], numIterations: Int, maxDepth: Int, learningRate: Double): GradientBoostedTreesModel = {
    val data = point_seed ++ point_samp
    var boostingStrategy = BoostingStrategy.defaultParams("Regression") //"Regression"
    boostingStrategy.setNumIterations(numIterations) //Note: Use more iterations in practice.
    boostingStrategy.setLearningRate(learningRate)
    boostingStrategy.treeStrategy.setMaxDepth(maxDepth)
    boostingStrategy.treeStrategy.numClasses
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training start*****************************")
    val model = GradientBoostedTrees.train(data, boostingStrategy)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************training end*****************************")
    val scoreAndLabels = data.map { point =>
      val prediction = model.predict(point.features)
      (prediction, point.label)
    }
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
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    var auc = metrics.areaUnderROC()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************auc = " + auc + "*****************************")

    scoreAndLabels.unpersist()
    //    println("Learned regression GBT model:\n" + model.toDebugString)
    model

  }

  def pre(model: GradientBoostedTreesModel, user_feature_forpre: RDD[(Long, DenseVector)], targetSize: Int): Array[(Long, Double)] = {
    val user_pre = user_feature_forpre.mapPartitions(iter => for (r <- iter) yield {
      val prediction = model.predict(r._2)
      (r._1, prediction)
    })
    val array = user_pre.collect().sortWith(_._2 > _._2)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************lookalike imei size = " + array.length + "*****************************")
    if (targetSize > 0) {
      array.take(targetSize)
    } else {
      array
    }


  }

  def load_data_onehot(table_in: String, uxip_boot_users_cycle: Int): RDD[(Long, Array[Double], String)] = {
    val c1 = Calendar.getInstance()
    c1.add(Calendar.DATE, -1)
    val date1 = sdf_date.format(c1.getTime())
    val sql_0 = "select max(stat_date) from algo.lookalike_feature_onehot"
    val date2 = hiveContext.sql(sql_0).map(r => r.getString(0)).collect()(0)
    val sql_1 = "select imei,feature,uxip_boot_users_cycle from " + table_in + " where stat_date=" + Math.min(date1.toLong, date2.toLong)
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    df.repartition(partition_num)
    val feature_rdd = df.mapPartitions(iter => for (r <- iter) yield {
      (r.getLong(0), r.getString(1).split(",").map(r1 => r1.toDouble), r.getString(2))
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
