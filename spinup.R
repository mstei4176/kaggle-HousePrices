# Date: April 6 2019
# Spark Version: 2.4.0
# Rsparkling Version: 0.2.22
# Sparkling Water Version. 2.4.9
# H2O R Version: 3.24.0.1
#######################################################################################################
library(sparklyr)
options(scipen=999)
options(encoding = "UTF-8")
Sys.setenv(TZ='UTC')
options(rsparkling.sparklingwater.version = "2.4.9")
library(rsparkling)
#######################################################################################################
config <- list(
  "sparklyr.shell.driver-memory"= "10g",
  #"sparklyr.shell.num-executors" = 3,
  "sparklyr.shell.executor-memory" = "22g",
  "sparklyr.shell.executor-cores" = 8,
  "sparklyr.sanitize.column.names" = "TRUE",
  "spark.ext.h2o.backend.cluster.mode" = "internal",
  "spark.ext.h2o.nthreads" = 8
  #"spark.ext.h2o.cluster.size" = 20
)
config[["spark.r.command"]] <- "/usr/local/bin/Rscript"
config$spark.ext.h2o.cloud.name <- "Bend.ai"
config$spark.executor.cores <- 8
#######################################################################################################
sc <- spark_connect("spark://10.0.1.106:7077",
                    spark_home = "/usr/local/share/spark-2.4.0-bin-hadoop2.7", 
                    app_name = "sparklyr",
                    version = "2.4.0",
                    config = config )
#######################################################################################################
library(h2o)
h2o_context(sc)
#######################################################################################################
                              
                              
                        



