package com.test
import java.io.{BufferedReader, InputStreamReader}
import java.text.SimpleDateFormat
import java.util.Date

import org.apache.hadoop.fs.Path
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.hive.HiveContext

import scala.collection.mutable.ArrayBuffer

/**
  * Created by xueyuan on 2017/7/28.检验种子人群的质量
  */
object check_seed {
  var sc: SparkContext = null
  var hiveContext: HiveContext = null
  //  val seed_file = "/tmp/xueyuan/seed1.txt"
  val sdf_date: SimpleDateFormat = new SimpleDateFormat("yyyyMMdd")
  val sdf_time: SimpleDateFormat = new SimpleDateFormat("HH:mm:ss")

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

    var imeiFile = ""
    var querysql = ""
    //新车    select  imei,fcate_qc  from user_profile.idl_fdt_dw_tag where  fcate_qc like '%,843702,%'
    println("***********************args_size=" + args.length + "*****************************")
    println("***********************args=" + args.mkString(",") + "*****************************")

    if (args.length == 2) {
      imeiFile = args(0)
      println("***********************imeiFile=" + imeiFile + "*****************************")
      querysql = args(1)
      println("***********************querysql=" + querysql + "*****************************")

    }
    //user
    val user = load_data(querysql)
    //seed
    val seed = get_seed(imeiFile)
    val seed_size = seed.length
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************seed_size=" + seed_size + "*****************************")
    val user_filter = user.filter(r => seed.contains(r._1))
    val user_filter_size = user_filter.count()
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************user_filter_size=" + user_filter_size + "*****************************")
    println(sdf_time.format(new Date((System.currentTimeMillis()))) + "***********************rate=" + (user_filter_size.toDouble / seed_size) + "*****************************")
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


  def load_data(querysql: String) = {
    val df = hiveContext.sql(querysql).map(r => (r.getLong(0).toString, r.getString(1)))
    println("***********************load_data finished" + df.count() + "*****************************")
    df
  }
}
