package com.alibaba.alink.pipeline.recommendation;

import java.util.function.Consumer;

import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.FmClassifierTrainBatchOp;
import com.alibaba.alink.operator.batch.dataproc.JsonValueBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalBinaryClassBatchOp;
import com.alibaba.alink.operator.batch.source.MemSourceBatchOp;

import com.alibaba.alink.operator.common.deepfm.DeepFmModelInfo;
import com.alibaba.alink.operator.common.deepfm.DeepFmPredictBatchOp;
import com.alibaba.alink.pipeline.classification.DeepFmClassifier;
import com.alibaba.alink.pipeline.classification.DeepFmModel;

// deep part

//TODO: add DeepFm regression
//import com.alibaba.alink.pipeline.regression.FmRegressor;

import org.junit.Test;

/**
 * @author Yuefeng Zhang
 * @date
 */
public class DeepFmTest {

    @Test
    public void testClassification() throws Exception {
        BatchOperator trainData = new MemSourceBatchOp(
                new Object[][] {
                        {"0:1.1 5:2.0", 1.0},
                        {"1:2.1 6:3.1", 1.0},
                        {"2:3.1 7:2.2", 1.0},
                        {"3:1.2 8:3.2", 0.0},
                        {"4:1.2 9:4.2", 0.0}
                },
                new String[] {"vec", "label"});
        DeepFmClassifier adagrad = new DeepFmClassifier()
                .setVectorCol("vec")
                .setLabelCol("label")
                .setNumEpochs(10)
                .setNumFactor(5)
                .setInitStdev(0.01)
                .setLearnRate(0.1)
                .setEpsilon(0.0001)
                .setLayers(new int[]{6, 5, 4})      // hidden layers' sizes
                .setPredictionCol("pred")
                .setPredictionDetailCol("details")
                .enableLazyPrintModelInfo();

        DeepFmModel model = adagrad.fit(trainData);
        BatchOperator result = model.transform(trainData);

        result.print();
    }
}
