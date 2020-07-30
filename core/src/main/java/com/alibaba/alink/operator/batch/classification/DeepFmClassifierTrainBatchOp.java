package com.alibaba.alink.operator.batch.classification;

import com.alibaba.alink.operator.common.deepfm.DeepFmTrainBatchOp;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;

import org.apache.flink.ml.api.misc.param.Params;

/**
 * DeepFm classification trainer.
 */
public class DeepFmClassifierTrainBatchOp extends DeepFmTrainBatchOp<DeepFmClassifierTrainBatchOp>
    implements DeepFmTrainParams<DeepFmClassifierTrainBatchOp> {
    private static final long serialVersionUID = -5053887118051783496L;

    /**
     * construct function.
     */
    public DeepFmClassifierTrainBatchOp() {
        super(new Params(), "binary_classification");
    }

    /**
     * construct function.
     *
     * @param params parameters needed by training process.
     */
    public DeepFmClassifierTrainBatchOp(Params params) {
        super(params, "binary_classification");
    }
}
