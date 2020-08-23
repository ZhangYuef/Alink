package com.alibaba.alink.pipeline.recommendation;

import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.DeepFmClassifierTrainBatchOp;
import com.alibaba.alink.operator.batch.dataproc.JsonValueBatchOp;
import com.alibaba.alink.operator.batch.dataproc.SplitBatchOp;
import com.alibaba.alink.operator.batch.dataproc.vector.VectorAssemblerBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalBinaryClassBatchOp;
import com.alibaba.alink.operator.batch.feature.OneHotPredictBatchOp;
import com.alibaba.alink.operator.batch.feature.OneHotTrainBatchOp;
import com.alibaba.alink.operator.batch.sink.CsvSinkBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.common.deepfm.DeepFmPredictBatchOp;
import com.alibaba.alink.operator.common.deepfm.DeepFmTrainBatchOp;
import org.junit.Test;

import java.io.File;

public class DeepFmDatasetTest {

    @Test
    public void pipelineTestBatch() throws Exception {

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

            String[] binaryCols = new String[] {"marital_status",  "occupation", "relationship", "race",
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

        DeepFmClassifierTrainBatchOp adagrad = new DeepFmClassifierTrainBatchOp()
                .setVectorCol("vec")
                .setLabelCol("label")
                .setNumEpochs(10)
                .setNumFactor(5)
                .setInitStdev(0.01)
                .setLearnRate(0.1)
                .setEpsilon(0.0001)
                .setLayers(new int[]{5, 5, 5})      // hidden layers' sizes
                .linkFrom(trainData);

        BatchOperator predictResult = new DeepFmPredictBatchOp().setVectorCol("vec").setPredictionCol("pred")
                .setPredictionDetailCol("details")
                .linkFrom(adagrad, testData);

        predictResult
                .link(
                        new EvalBinaryClassBatchOp()
                                .setLabelCol("label")
                                .setPredictionDetailCol("details")
                )
                .link(
                        new JsonValueBatchOp()
                                .setSelectedCol("Data")
                                .setReservedCols(new String[]{"Statistics"})
                                .setOutputCols(new String[]{"Accuracy", "AUC", "ConfusionMatrix"})
                                .setJsonPath(new String[]{"$.Accuracy", "$.AUC", "$.ConfusionMatrix"})
                )
                .print();

    }
}
