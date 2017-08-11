package com.test

import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util
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
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/6/15.
  */
object getnerate_feature_by_gbdt {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  val seed_file1 = "/tmp/xueyuan/seed1.txt"
  val model_path = "/tmp/xueyuan/gbdt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  var strategy = "Regression"
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
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    val uxip_boot_users_cycle = 7
    val table_in = "algo.lookalike_feature_onehot"
    val table_out ="algo.xueyuan_lookalike_gbdtfeature"
    val user_feature = load_data_onehot(table_in)
    val user_feature_cyc = user_feature.filter(r => (r._3.contains("," + uxip_boot_users_cycle + ","))).mapPartitions(iter => for (r <- iter) yield (r._1, r._2))
    val splits = user_feature_cyc.randomSplit(Array(0.5, 0.5), seed = 11L)
    val posi = splits(0).mapPartitions(iter => for (r <- iter) yield new LabeledPoint(1.0, r._2))
    val nega = splits(1).mapPartitions(iter => for (r <- iter) yield new LabeledPoint(0.0, r._2))
    //training
    val model = training_reg(posi, nega)
    //    val model = GradientBoostedTreesModel.load(sc, model_path)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************model finished *****************************")
    // generate feature
    val trees = model.trees

    val user_treeid_score = user_feature_cyc.mapPartitions(iter => {
      for (r <- iter) yield {
        val vector = r._2
        val result = new ArrayBuffer[(Int, Double)]()
        for (i <- 0 until trees.length) {
          result += ((i, trees(i).predict(vector))) //(treeid,pre_score)
        }
        (r._1, result.toArray)
      }

    })
    user_treeid_score.cache()
    user_treeid_score.repartition(partition_num)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************user_treeid_score_label=" + user_treeid_score.count() + " ****************************")
    val treeid_scoreset = user_treeid_score.flatMap(r => r._2).mapPartitions(iter => for (r <- iter) yield (r._1, new HashSet[Double]() + r._2)).reduceByKey(_ ++ _)
    treeid_scoreset.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************treeid_scoreset=" + treeid_scoreset.count() + " ****************************")
    //100
    val treeid_score_scoreindex = treeid_scoreset.mapPartitions(iter => for (r <- iter) yield {
      var score_array = r._2.toArray
      var map = score_array.zipWithIndex
      (r._1, map)
    })
    treeid_score_scoreindex.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************treeid_score_scoreindex=" + treeid_score_scoreindex.count() + " ****************************")
    treeid_scoreset.unpersist()
    val treeid_score_gbdtfeature = treeid_score_scoreindex.mapPartitions(iter => for (r <- iter) yield {
      var treeid_score_gbdtfeaturearray = new ArrayBuffer[((Int, Double), Array[Int])]()
      for ((score, scoreindex) <- r._2) {
        val gbdtfeature = new Array[Int](r._2.size)
        gbdtfeature(scoreindex) = 1
        val temp = (((r._1, score), gbdtfeature))
        treeid_score_gbdtfeaturearray += temp
      }
      treeid_score_gbdtfeaturearray.toArray
    }).flatMap(r => r).collect()


    val treeid_score_gbdtfeature_br = sc.broadcast(treeid_score_gbdtfeature.toMap)
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************treeid_score_gbdtfeature=" + treeid_score_gbdtfeature.length + " ****************************")
    treeid_score_scoreindex.unpersist()
    val user_onehot = user_treeid_score.mapPartitions(iter => for (r <- iter) yield {
      val map = treeid_score_gbdtfeature_br.value
      var onehot = new ArrayBuffer[Int]()
      val gbdtfeature = r._2
      for (g <- gbdtfeature) {
        val temp = map(g)
        onehot ++= temp
      }
      (r._1, onehot.toArray)
    })
    user_onehot.cache()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************user_onehot_label=" + user_onehot.count() + " ****************************")



      save_data(table_out, user_onehot)


  }

  def get_seed(table_in: String): Array[String] = {
    //    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    //    val path1 = new Path(seed_file1)
    //    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
    //    var seed_id = new ArrayBuffer[String]()
    //    var line1 = reader1.readLine()
    //    while (line1 != null) {
    //      if (!line1.equals("null")) {
    //        seed_id += line1.trim
    //      }
    //      line1 = reader1.readLine()
    //    }
    //    seed_id.toArray
    val sql_1 = "select imei from " + table_in
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished*****************************")
    df.repartition(partition_num)
    val seed_rdd = df.mapPartitions(iter => for (r <- iter) yield {
      r.getString(0)
    })
    seed_rdd.collect()
  }


  def get_lable_feature_rdd(user_feature: RDD[(Long, Array[Double], Int)]): RDD[LabeledPoint] = {
    val lable_feature_rdd = user_feature.map(r => {
      val dv = new DenseVector(r._2)
      (r._3, dv)
    })
    lable_feature_rdd.map(r => new LabeledPoint(r._1, r._2))
  }

  def training_reg(point_seed: RDD[LabeledPoint], point_samp: RDD[LabeledPoint]): GradientBoostedTreesModel = {
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
      val feature = r.getString(2)
      val cyc = r.getString(3)
      var feature_index = Array.empty[Int]
      if (feature != null && feature.length > 0) {
        feature_index = feature.split(",").map(index => index.toInt)
      }
      val sp = new SparseVector(r.getInt(1), feature_index, Array.fill(feature_index.length)(1.0))
      (imei, sp, cyc)
    })
    feature_rdd
  }

  def save_data(table_out: String, user_onehot: RDD[(Long, Array[Int])]): Unit = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save table start*****************************")
    val user_onehot_forsave = user_onehot.map(r => {
      var onehot = ""
      for (i <- r._2) {
        onehot += i + ","
      }
      if (onehot.length > 2) {
        onehot = onehot.substring(0, onehot.length - 1)
      }
      (r._1, onehot)
    })
    val candidate_rdd = user_onehot_forsave.map(r => Row(r._1, r._2))

    val structType = StructType(
      StructField("imei", LongType, false) ::
        StructField("feature", StringType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( imei bigint,feature string ) partitioned by (stat_date bigint) stored as textfile"
    val c1 = Calendar.getInstance()
    //    c1.add(Calendar.DATE, -1)
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
