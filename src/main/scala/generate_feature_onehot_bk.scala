package com.test

import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util.{Calendar, Date}

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.hive.HiveContext
import org.apache.spark.sql.types._
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.functions.udf

import scala.collection.mutable.ArrayBuffer
import scala.collection.{Map, mutable}

/**
  * Created by xueyuan on 2017/7/21.增加任意特征，预处理为onehot编码存入数据库
  */
object generate_feature_onehot_bk {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  var sqlContext: SQLContext = null
  //  val seed_file = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")
  val partition_num = 200


  val oriFeatureMap = new mutable.HashMap[String, Int]()

  def main(args: Array[String]): Unit = {
    val userName = "mzsip"
    System.setProperty("user.name", userName)
    System.setProperty("HADOOP_USER_NAME", userName)
    println("***********************start*****************************")
    val sparkConf: SparkConf = new SparkConf().setAppName("xueyuan_lookalike")
    sc = new SparkContext(sparkConf)
    println("***********************sc*****************************")
    sc.hadoopConfiguration.set("mapred.output.compress", "false")
    hiveContext = new HiveContext(sc)
    sqlContext = new SQLContext(sc)
    println("***********************hive*****************************")
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    //feature列名，文件中每行一个列名
    var test = false
    var feature_col_file = "/tmp/xueyuan/online/lookalike/feature_col.txt"
    var table_out = ""
    if (args.length == 3) {
      test = args(0).toBoolean
      println("***********************test=" + test + "*****************************")
      feature_col_file = args(1)
      println("***********************feature_col_file=" + feature_col_file + "*****************************")
      table_out = args(2)
      println("***********************table_out=" + table_out + "*****************************")
    }

    var feature: Array[String] = null
    var df: DataFrame = null
    if (test) {
      feature = Array("sex", "marriage_status")
      df = {
        sqlContext.createDataFrame(Seq(
          (0, 3, "male,other", 5),
          (1, 7, "female", 6),
          (2, 7, "female", 5),
          (3, 7, "female", 6)
        )).toDF("imei", "uxip_boot_users_cycle", "sex", "marriage_status")
      }
    } else {
      feature = load_feature(feature_col_file)
      df = load_data(feature)
    }
    //将所有feature转为index
    for (col <- feature) {
      df = oneColProcess(col)(df)
    }
    if (test) {
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature index*****************************")
      for (row <- df.take(10)) {
        println(row.toString())
      }

    }
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************oriFeatureMap*****************************")
    var sum = 0
    for ((k, v) <- oriFeatureMap) {
      println(k + ":" + v)
      sum = sum + v
    }
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************oriFeatureMap sum=" + sum + "*****************************")
    //转换为onehot编码
    for (col <- feature) {
      val catSize = oriFeatureMap(col)
      df = oneColProcessWithOneHot(col, catSize)(df)
    }
    if (test) {
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************feature onhot*****************************")
      for (row <- df.take(10)) {
        println(row.toString())
      }
    }


    val res = df.map(r => {
      val imei = r.getAs("imei").toString.toLong
      val uxip_boot_users_cycle = r.getAs("uxip_boot_users_cycle").toString
      val app_cur_com_meizu_media_reader = r.getAs("app_cur_com_meizu_media_reader").toString
      val feature_value_map = r.getValuesMap[String](feature)
      var new_feature = new ArrayBuffer[String]()
      for (f <- feature) {
        val v = feature_value_map(f)
        if (v != null && v.length > 0) {
          new_feature += v
        }
      }
      val feature_array = new_feature.mkString(",").split(",").map(f => f.toDouble)
      val size = feature_array.size
      val index_array = new ArrayBuffer[Int]()
      for (i <- 0 until feature_array.length) {
        if (feature_array(i) == 1.0) {
          index_array += i
        }
      }
      (imei, size, index_array.mkString(","), uxip_boot_users_cycle, app_cur_com_meizu_media_reader)
    })
    if (test) {
      println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************rdd*****************************")
      for ((imei, size, cyc, fea, reader) <- res.take(10)) {
        println(imei + " " + size + " " + cyc + " " + fea)
      }
    }
    save_feature(table_out, res)

  }

  def save_feature(table_out: String, data: RDD[(Long, Int, String, String, String)]): Unit = {
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************save table start*****************************")
    val candidate_rdd = data.map(r => Row(r._1, r._2, r._3, r._4, r._5))

    val structType = StructType(
      StructField("imei", LongType, false) ::
        StructField("feature_size", IntegerType, false) ::
        StructField("feature", StringType, false) ::
        StructField("uxip_boot_users_cycle", StringType, false) ::
        StructField("app_cur_com_meizu_media_reader", StringType, false) ::
        Nil
    )
    //from RDD to DataFrame
    val candidate_df = hiveContext.createDataFrame(candidate_rdd, structType)
    val create_table_sql: String = "create table if not exists " + table_out + " ( imei bigint,feature_size int, feature string,uxip_boot_users_cycle string,app_cur_com_meizu_media_reader string ) partitioned by (stat_date bigint) stored as textfile"
    val c1 = Calendar.getInstance()
    //        c1.add(Calendar.DATE, -1)
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

  def load_feature(feature_col_file: String) = {
    val hdfs = org.apache.hadoop.fs.FileSystem.get(new org.apache.hadoop.conf.Configuration())
    val path1 = new Path(feature_col_file)
    val reader1 = new BufferedReader(new InputStreamReader(hdfs.open(path1), "utf-8"))
    var feature = new ArrayBuffer[String]()
    var line1 = reader1.readLine()
    while (line1 != null) {
      val f = line1.trim
      if (!f.equals("null") && f.length > 0) {
        feature += f
      }
      line1 = reader1.readLine()
    }
    feature.toArray
  }

  def load_data(feature: Array[String]) = {
    val feature_string = feature.mkString(",")
    val feature_num = feature.length
    val sql_1 = "select imei,uxip_boot_users_cycle," + feature_string + ",app_cur_com_meizu_media_reader from user_profile.idl_fdt_dw_tag"
    val df = hiveContext.sql(sql_1)
    println("***********************load_data finished" + df.count() + "*****************************")
    df
  }

  def oneColProcessWithOneHot(col: String, catSize: Int) = (df: DataFrame) => {
    val stringToVector = udf[String, String] { w =>
      if (w != null && w.length > 0) {
        var index_arr = new ArrayBuffer[Int]()
        for (ele <- w.split(",")) {
          if (ele != null && ele.length > 0) {
            //排除空字符的影响
            index_arr += ele.toInt
          }
        }
        val value_arr = Array.fill(index_arr.length)(1.0)
        val vec = Vectors.sparse(catSize, index_arr.toArray, value_arr)
        val arr = vec.toArray
        arr.mkString(",")
      } else {
        Array.fill(catSize)(0.0).mkString(",")
      }
    }
    df.withColumn(col, stringToVector(df(col)))
  }

  def oneColProcess(col: String) = (df: DataFrame) => {
    val sma = df.schema
    sma(col).dataType match {
      case StringType => {
        //        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        val catMap2 = df.select(col).flatMap(_.getString(0).split(",")).distinct.filter(!"".equals(_)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap2.size
        val stringToDouble = udf[String, String] { w =>
          val arr = w.split(",").filter(!"".equals(_))
          var res = new ArrayBuffer[Int]()
          for (ele <- arr) {
            val r = catMap2(ele)
            res += r
          }
          res.toArray.mkString(",")
          //          catMap(_)
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
      case LongType => {
        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap.size
        val stringToDouble = udf[String, Long] { w =>
          Array(catMap(w)).mkString(",")
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
      case DoubleType => {
        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap.size
        val stringToDouble = udf[String, Double] { w =>
          Array(catMap(w)).mkString(",")
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
      case IntegerType => {
        val catMap = df.select(col).distinct.map(_.get(0)).collect.zipWithIndex.toMap
        oriFeatureMap(col) = catMap.size
        val stringToDouble = udf[String, Int] { w =>
          Array(catMap(w)).mkString(",")
        }
        df.withColumn(col, stringToDouble(df(col)))
      }
    }
  }
}
