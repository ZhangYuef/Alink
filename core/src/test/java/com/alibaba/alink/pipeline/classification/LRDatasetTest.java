package com.alibaba.alink.pipeline.classification;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import com.alibaba.alink.common.MLEnvironmentFactory;
import com.alibaba.alink.operator.AlgoOperator;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.LogisticRegressionPredictBatchOp;
import com.alibaba.alink.operator.batch.classification.LogisticRegressionTrainBatchOp;
import com.alibaba.alink.operator.batch.dataproc.JsonValueBatchOp;
import com.alibaba.alink.operator.batch.dataproc.SplitBatchOp;
import com.alibaba.alink.operator.batch.dataproc.vector.VectorAssemblerBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalBinaryClassBatchOp;
import com.alibaba.alink.operator.batch.feature.OneHotPredictBatchOp;
import com.alibaba.alink.operator.batch.feature.OneHotTrainBatchOp;
import com.alibaba.alink.operator.batch.sink.CsvSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.batch.source.MemSourceBatchOp;
import com.alibaba.alink.operator.stream.StreamOperator;
import com.alibaba.alink.operator.stream.source.MemSourceStreamOp;
import com.alibaba.alink.pipeline.Pipeline;
import com.alibaba.alink.pipeline.PipelineModel;

import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;
import org.junit.Assert;
import org.junit.Test;

public class LRDatasetTest {

    @Test
    public void pipelineTestBatch() throws Exception {
        MLEnvironmentFactory.getDefault().getExecutionEnvironment().setParallelism(1);

        String trainPath = "/Users/zyf/Desktop/AdultDataset/adult-strip-train-save.data";
        File f = new File(trainPath);
        if (!f.exists()) {
            BatchOperator data = new CsvSourceBatchOp()
                    .setFilePath("/Users/zyf/Desktop/AdultDataset/adult-strip.data")
                    .setSchemaStr("age double,workclass string,fnlwgt double,education string,education_num double,"
                            + " marital_status string,occupation string,relationship string,"
                            + "race string,sex string, capital_gain double,"
                            + "capital_loss double,hours_per_week double,native_country string,label string");

            String[] featureColNames
                    = new String[] {"age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week",
                    "vec"};

            String[] binaryCols = new String[] {"marital_status", "occupation", "relationship", "race",
                    "sex", "education", "workclass", "native_country"};

            BatchOperator oneHot = new OneHotTrainBatchOp().setSelectedCols(binaryCols).linkFrom(data);

            BatchOperator normalData = new OneHotPredictBatchOp().setOutputCols("vec").linkFrom(oneHot, data);

            normalData = normalData.link(new VectorAssemblerBatchOp().setSelectedCols(featureColNames)
                    .setOutputCol("vec").setReservedCols("label"));

            normalData.link(new CsvSinkBatchOp().setFilePath(trainPath).setOverwriteSink(true));
            BatchOperator.execute();
        }

        // Read data from file.
        BatchOperator allData = new CsvSourceBatchOp()
                .setFilePath(trainPath)
                .setSchemaStr("label string, vec string");

        // Train, test data split.
        SplitBatchOp spliter = new SplitBatchOp().setFraction(0.8);
        spliter.linkFrom(allData);

        BatchOperator trainData = spliter;
        BatchOperator testData = spliter.getSideOutput(0);

        LogisticRegressionTrainBatchOp trainLr = new LogisticRegressionTrainBatchOp()
                .setVectorCol("vec")
                .setLabelCol("label")
                .setMaxIter(100)
                .setOptimMethod("Owlqn")
                .setWithIntercept(true)
                .setStandardization(true)
                .setL1(0.001)
                .setEpsilon(1.0e-6)
                .linkFrom(trainData);//.print();
        // trainLr.lazyPrintTrainInfo();
        //BatchOperator trainLr = new LogisticRegressionTrainBatchOp(new Params()
        //    .set("vectorCol", "vector")
        //    .set("labelColName", labelColName)
        //    .set("maxIter", 500)).setEpsilon(1.0e-7)
        //    .setOptimMethod("LBFGS")
        //    .linkFrom(normalData);

        Params predParams = new Params();
        // Use test data to do prediction.
        BatchOperator result = new LogisticRegressionPredictBatchOp(predParams).setVectorCol("vec")
                .setPredictionCol("pred")
                .setPredictionDetailCol("details")
                .linkFrom(trainLr, testData);

        new EvalBinaryClassBatchOp()
                .setLabelCol("label")
                .setPredictionDetailCol("details")
                .setPositiveLabelValueString("<=50K")
                .linkFrom(result)
                .link(new JsonValueBatchOp()
                        .setSelectedCol("Data")
                        .setReservedCols(new String[]{"Statistics"})
                        .setOutputCols(new String[]{"Accuracy", "AUC"})
                        .setJsonPath(new String[]{"$.Accuracy", "$.AUC"}))
                .print();
    }
}
