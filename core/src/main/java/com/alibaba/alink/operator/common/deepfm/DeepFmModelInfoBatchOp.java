package com.alibaba.alink.operator.common.deepfm;

import java.util.List;

import com.alibaba.alink.common.lazy.ExtractModelInfoBatchOp;
import com.alibaba.alink.operator.batch.BatchOperator;

import com.alibaba.alink.pipeline.classification.DeepFmModel;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;

/**
 * DeepFmModelInfoBatchOp can be linked to the output of BaseDeepFmTrainBatchOp to summary the DeepFm model.
 */
public class DeepFmModelInfoBatchOp
    extends ExtractModelInfoBatchOp<DeepFmModelInfo, DeepFmModelInfoBatchOp> {
    private TypeInformation labelType;

    public DeepFmModelInfoBatchOp(TypeInformation labelType) {
        this(new Params());
        this.labelType = labelType;
    }

    /**
     * construct function.
     * @param params - model parameters
     */
    public DeepFmModelInfoBatchOp(Params params) {
        super(params);
    }

    @Override
    protected DeepFmModelInfo createModelInfo(List<Row> rows) {
        return new DeepFmModelInfo(rows, labelType);
    }

    @Override
    protected BatchOperator<?> processModel() {
        return this;
    }

}
