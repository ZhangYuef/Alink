package com.alibaba.alink.pipeline.classification;

import com.alibaba.alink.common.MLEnvironmentFactory;
import com.alibaba.alink.common.utils.httpsrc.Iris;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.dataproc.SplitBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalMultiClassBatchOp;
import com.alibaba.alink.operator.common.evaluation.MultiClassMetrics;
import org.junit.Assert;
import org.junit.Test;

public class MultilayerPerceptronClassifierTest {

    @Test
    public void testMLPC() throws Exception {
        MLEnvironmentFactory.getDefault().getExecutionEnvironment().setParallelism(1);
        BatchOperator data = Iris.getBatchData();

        // Train, test data split.
        SplitBatchOp spliter = new SplitBatchOp().setFraction(0.8);
        spliter.linkFrom(data);

        BatchOperator trainData = spliter;
        BatchOperator testData = spliter.getSideOutput(0);

        MultilayerPerceptronClassifier classifier = new MultilayerPerceptronClassifier()
                .setFeatureCols(Iris.getFeatureColNames())
                .setLabelCol(Iris.getLabelColName())
                .setLayers(new int[]{4, 5, 3})
                //.setDropoutRate(0.5)
                .setMaxIter(10)
                .setPredictionCol("pred_label")
                .setPredictionDetailCol("pred_detail");

        BatchOperator res = classifier.fit(trainData).transform(testData);

        MultiClassMetrics metrics = new EvalMultiClassBatchOp()
            .setPredictionDetailCol("pred_detail")
            .setLabelCol(Iris.getLabelColName())
            .linkFrom(res)
            .collectMetrics();

        System.out.println(metrics.getAccuracy());
//        Assert.assertTrue(metrics.getAccuracy() > 0.6);
    }
}