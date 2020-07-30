package com.alibaba.alink.pipeline.classification;

import com.alibaba.alink.operator.common.deepfm.DeepFmModelMapper;
import com.alibaba.alink.params.recommendation.DeepFmPredictParams;
import com.alibaba.alink.pipeline.MapModel;

import org.apache.flink.ml.api.misc.param.Params;


public class DeepFmModel extends MapModel<DeepFmModel>
        implements DeepFmPredictParams<DeepFmModel> {

    private static final long serialVersionUID = -1751100576503606882L;

    public DeepFmModel() {
        this(null);
    }

    public DeepFmModel(Params params) {
        super(DeepFmModelMapper::new, params);
    }

}
