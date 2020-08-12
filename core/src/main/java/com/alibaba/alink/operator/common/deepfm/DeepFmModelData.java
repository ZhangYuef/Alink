package com.alibaba.alink.operator.common.deepfm;

import java.io.Serializable;

import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.DeepFmDataFormat;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.Task;

/**
 * DeepFM model data for {@link com.alibaba.alink.pipeline.classification.DeepFmClassifier}.
 */
public class DeepFmModelData implements Serializable {
    private static final long serialVersionUID = 5575657492265170833L;

    public String vectorColName = null;
    public String[] featureColNames = null;
    public String labelColName = null;
    public DeepFmDataFormat deepFmModel;
    public int vectorSize;
    public int[] dim;
    public int[] fieldPos;
    public Object[] labelValues = null;
    public Task task;
}
