package com.alibaba.alink.pipeline.classification;

import breeze.linalg.$times$;
import com.alibaba.alink.common.lazy.HasLazyPrintModelInfo;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.classification.DeepFmClassifierTrainBatchOp;

import com.alibaba.alink.params.recommendation.DeepFmPredictParams;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;
import com.alibaba.alink.pipeline.Trainer;

import org.apache.flink.ml.api.misc.param.Params;

/**
 * DeepFm classifier pipeline op.
 *
 */
public class DeepFmClassifier extends Trainer <DeepFmClassifier, DeepFmModel>
    implements DeepFmTrainParams<DeepFmClassifier>, DeepFmPredictParams<DeepFmClassifier>, HasLazyPrintModelInfo<DeepFmClassifier> {
    private static final long serialVersionUID = 8803336211233701463L;

    /**
     * construct function.
     *
     */
    public DeepFmClassifier() {
        super();
    }

    /**
     * construct function.
     *
     * @param params parameters needed by training process.
     */
    public DeepFmClassifier(Params params) {
        super(params);
    }

    @Override
    protected BatchOperator train(BatchOperator in) {
        return new DeepFmClassifierTrainBatchOp(this.getParams()).linkFrom(in);
    }
}
