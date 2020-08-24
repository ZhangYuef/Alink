package com.alibaba.alink.operator.common.deepfm;

import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.mapper.RichModelMapper;
import com.alibaba.alink.common.utils.JsonConverter;
import com.alibaba.alink.common.utils.TableUtil;

import com.alibaba.alink.operator.common.linear.FeatureLabelUtil;
import com.alibaba.alink.operator.common.optim.DeepFmOptimizer;

import com.alibaba.alink.operator.common.utils.FmOptimizerUtils;
import com.alibaba.alink.params.classification.SoftmaxPredictParams;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * DeepFm mapper maps one sample to a sample with a predicted label.
 */
public class DeepFmModelMapper extends RichModelMapper {

    private static final long serialVersionUID = -6348182372481296494L;
    private int vectorColIndex = -1;
    private int[] featIdx = null;
    private int featLen = -1;
    private DeepFmModelData model;
    private int[] dim;

    public DeepFmModelMapper(TableSchema modelSchema, TableSchema dataSchema, Params params) {
        super(modelSchema, dataSchema, params);
        if (null != params) {
            String vectorColName = params.get(SoftmaxPredictParams.VECTOR_COL);
            if (null != vectorColName && vectorColName.length() != 0) {
                this.vectorColIndex = TableUtil.findColIndexWithAssert(dataSchema.getFieldNames(), vectorColName);
            }
        }
    }

    @Override
    public void loadModel(List<Row> modelRows) {
        DeepFmModelDataConverter deepFmModelDataConverter
                = new DeepFmModelDataConverter(DeepFmModelDataConverter.extractLabelType(super.getModelSchema()));
        this.model = deepFmModelDataConverter.load(modelRows);
        this.dim = model.dim;

        if (vectorColIndex == -1) {
            TableSchema dataSchema = getDataSchema();
            if (this.model.featureColNames != null) {
                this.featIdx = new int[this.model.featureColNames.length];
                featLen = featIdx.length;
                String[] predictTableColNames = dataSchema.getFieldNames();
                for (int i = 0; i < this.featIdx.length; i++) {
                    this.featIdx[i] = TableUtil.findColIndexWithAssert(predictTableColNames,
                            this.model.featureColNames[i]);
                }
            } else {
                vectorColIndex = TableUtil.findColIndexWithAssert(dataSchema.getFieldNames(), model.vectorColName);
            }
        }
    }

    @Override
    protected Object predictResult(Row row) throws Exception {
        Vector vec = FeatureLabelUtil.getFeatureVector(row, false, featLen,
                this.featIdx, this.vectorColIndex, model.vectorSize);
        double y = FmOptimizerUtils.fmCalcY(vec, model.deepFmModel.linearItems, model.deepFmModel.factors,
                model.deepFmModel.bias, dim).f0 + DeepFmOptimizer.deepCalcY(model.deepFmModel, vec, dim).get(0);

        if (model.task.equals(BaseDeepFmTrainBatchOp.Task.REGRESSION)) {
            return y;
        } else if (model.task.equals(BaseDeepFmTrainBatchOp.Task.BINARY_CLASSIFICATION)) {
            y = logit(y);
            return (y <= 0.5 ? model.labelValues[1] : model.labelValues[0]);
        } else {
            throw new RuntimeException("This task is not supported.");
        }
    }

    @Override
    protected Tuple2<Object, String> predictResultDetail(Row row) throws Exception {
        Vector vec = FeatureLabelUtil.getFeatureVector(row, false, featLen,
                featIdx, this.vectorColIndex, model.vectorSize);
        double y = FmOptimizerUtils.fmCalcY(vec, model.deepFmModel.linearItems, model.deepFmModel.factors,
                model.deepFmModel.bias, dim).f0 + DeepFmOptimizer.deepCalcY(model.deepFmModel, vec, dim).get(0);

        if (model.task.equals(BaseDeepFmTrainBatchOp.Task.REGRESSION)) {
            String detail = String.format("{\"%s\":%f}", "label", y);
            return Tuple2.of(y, detail);
        } else if (model.task.equals(BaseDeepFmTrainBatchOp.Task.BINARY_CLASSIFICATION)) {
            y = logit(y);
            Object label = (y <= 0.5 ? model.labelValues[1] : model.labelValues[0]);
            Map<String, String> detail = new HashMap<>(0);
            detail.put(model.labelValues[1].toString(), Double.valueOf(1 - y).toString());
            detail.put(model.labelValues[0].toString(), Double.valueOf(y).toString());
            String jsonDetail = JsonConverter.toJson(detail);
            return Tuple2.of(label, jsonDetail);
        } else {
            throw new RuntimeException("This task is not supported.");
        }
    }

    private static double logit(double x) {
        return 1. / (1. + Math.exp(-x));
    }

    public DeepFmModelData getModel() {
        return model;
    }
}
