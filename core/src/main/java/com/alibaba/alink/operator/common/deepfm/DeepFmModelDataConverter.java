package com.alibaba.alink.operator.common.deepfm;

import java.util.Arrays;
import java.util.Collections;

import com.alibaba.alink.common.model.LabeledModelDataConverter;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.common.utils.JsonConverter;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.DeepFmDataFormat;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.Task;
import com.alibaba.alink.operator.common.linear.FeatureLabelUtil;

import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;

/**
 * DeepFm model converter. This converter can help serialize and deserialize the model data.
 */
public class DeepFmModelDataConverter extends LabeledModelDataConverter<DeepFmModelData, DeepFmModelData> {

    public DeepFmModelDataConverter() { this(null); }

    /**
     * @param labelType label type.
     */
    public DeepFmModelDataConverter(TypeInformation labelType) {
        super(labelType);
    }

    /**
     * Serialize deepFM model data.
     * @param modelData The model data to serialize.
     * @return The serialization result.
     */
    @Override
    protected Tuple3<Params, Iterable<String>, Iterable<Object>> serializeModel(DeepFmModelData modelData) {
        Params meta = new Params()
                .set(ModelParamName.VECTOR_COL_NAME, modelData.vectorColName)
                .set(ModelParamName.LABEL_COL_NAME, modelData.labelColName)
                .set(ModelParamName.TASK, modelData.task.toString())
                .set(ModelParamName.VECTOR_SIZE, modelData.vectorSize)
                .set(ModelParamName.FEATURE_COL_NAMES, modelData.featureColNames)
                .set(ModelParamName.LABEL_VALUES, modelData.labelValues)
                .set(ModelParamName.DIM, modelData.dim)
                .set(ModelParamName.FIELD_POS, modelData.fieldPos);
        DeepFmDataFormat factors = modelData.deepFmModel;

        return Tuple3.of(meta, Collections.singletonList(JsonConverter.toJson(factors)),
                Arrays.asList(modelData.labelValues));
    }

    /**
     * Deserialize deepFM model data.
     * @param meta           The model meta data.
     * @param data           The model concrete data.
     * @param distinctLabels Distinct label values of training data.
     * @return DeepFmModelData
     */
    @Override
    protected DeepFmModelData deserializeModel(Params meta, Iterable<String> data, Iterable<Object> distinctLabels) {
        DeepFmModelData modelData = new DeepFmModelData();

        if (meta.contains(ModelParamName.LABEL_VALUES)) {
            modelData.labelValues = FeatureLabelUtil.recoverLabelType(meta.get(ModelParamName.LABEL_VALUES),
                    this.labelType);
        }
        String json = data.iterator().next();
        modelData.deepFmModel = JsonConverter.fromJson(json, DeepFmDataFormat.class);
        modelData.vectorColName = meta.get(ModelParamName.VECTOR_COL_NAME);
        modelData.featureColNames = meta.get(ModelParamName.FEATURE_COL_NAMES);
        modelData.labelColName = meta.get(ModelParamName.LABEL_COL_NAME);
        modelData.task = Task.valueOf(meta.get(ModelParamName.TASK));
        modelData.dim = meta.get(ModelParamName.DIM);
        modelData.vectorSize = meta.get(ModelParamName.VECTOR_SIZE);
        modelData.fieldPos = meta.get(ModelParamName.FIELD_POS);

        if (meta.contains(ModelParamName.LABEL_VALUES)) {
            modelData.labelValues = FeatureLabelUtil.recoverLabelType(meta.get(ModelParamName.LABEL_VALUES),
                    this.labelType);
        }

        return modelData;
    }
}
