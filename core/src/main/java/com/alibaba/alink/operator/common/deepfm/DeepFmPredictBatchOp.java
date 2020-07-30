package com.alibaba.alink.operator.common.deepfm;

import com.alibaba.alink.operator.batch.classification.DeepFmClassifierPredictBatchOp;
import com.alibaba.alink.operator.batch.utils.ModelMapBatchOp;
import com.alibaba.alink.params.recommendation.DeepFmPredictParams;

import org.apache.flink.ml.api.misc.param.Params;

/**
 * DeepFm predict batch operator. This operator predicts data's label with DeepFm model.
 *
 */
public final class DeepFmPredictBatchOp extends ModelMapBatchOp <DeepFmPredictBatchOp>
    implements  DeepFmPredictParams<DeepFmPredictBatchOp>{
    private static final long serialVersionUID = 7744457450905884100L;

    /**
     * construct function.
     */
    public DeepFmPredictBatchOp() {
        this(new Params());
    }

    /**
     * construct function.
     *
     * @param params parameters needed by predicting process.
     */
    public DeepFmPredictBatchOp(Params params) {
        super(DeepFmModelMapper::new, params);
    }
}
