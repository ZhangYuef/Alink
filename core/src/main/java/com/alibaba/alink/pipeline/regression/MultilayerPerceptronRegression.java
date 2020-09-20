package com.alibaba.alink.pipeline.regression;

import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.MultilayerPerceptronTrainBatchOp;
import com.alibaba.alink.params.classification.MultilayerPerceptronPredictParams;
import com.alibaba.alink.params.classification.MultilayerPerceptronTrainParams;
import com.alibaba.alink.pipeline.Trainer;
import com.alibaba.alink.pipeline.classification.MultilayerPerceptronClassificationModel;
import com.alibaba.alink.pipeline.classification.MultilayerPerceptronClassifier;
import org.apache.flink.ml.api.misc.param.Params;

/**
 * MultilayerPerceptronRegression is a neural network based multi-class classifier.
 * Vanilla neural network with all dense layers are used, the output layer is a softmax layer.
 * Number of inputs has to be equal to the size of feature vectors.
 * Number of outputs has to be equal to the total number of labels.
 */
public class MultilayerPerceptronRegression
        extends Trainer <MultilayerPerceptronRegression, MultilayerPerceptronClassificationModel> implements
        MultilayerPerceptronTrainParams <MultilayerPerceptronRegression>,
        MultilayerPerceptronPredictParams <MultilayerPerceptronRegression> {

    public MultilayerPerceptronRegression() {
        super(new Params());
    }

    public MultilayerPerceptronRegression(Params params) {
        super(params);
    }

    @Override
    protected BatchOperator train(BatchOperator in) {
        return new MultilayerPerceptronTrainBatchOp(this.getParams()).linkFrom(in);
    }
}
