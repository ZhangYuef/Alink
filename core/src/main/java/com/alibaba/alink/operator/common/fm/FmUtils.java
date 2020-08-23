package com.alibaba.alink.operator.common.fm;

import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.SparseVector;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.linalg.VectorUtil;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.common.utils.TableUtil;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp;
import com.alibaba.alink.operator.common.deepfm.DeepFmModelData;
import com.alibaba.alink.operator.common.deepfm.DeepFmModelDataConverter;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;
import com.alibaba.alink.params.recommendation.FmTrainParams;
import com.alibaba.alink.pipeline.classification.FmModel;
import org.apache.commons.lang.NotImplementedException;
import org.apache.flink.api.common.functions.FilterFunction;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class FmUtils {

    /**
     * @param initData  (weight, object(index, label), vector), feature vector> get some useful info from initial data.
     * @param isRegProc train process is regression or classification.
     * @return useful data, including label values and vector size which represents for the largest index.
     */
    public static DataSet<Tuple2<Object[], Integer>> getUtilInfo(
            DataSet<Tuple3<Double, Object, Vector>> initData,
            boolean isRegProc) {
        return initData.filter(
                new FilterFunction<Tuple3<Double, Object, Vector>>() {
                    private static final long serialVersionUID = -8853068410321180715L;

                    @Override
                    public boolean filter(Tuple3<Double, Object, Vector> value) throws Exception {
                        if (value.f0 < 0.0) {
                            return true;
                        } else {
                            return false;
                        }
                    }
                }).reduceGroup(
                new GroupReduceFunction<Tuple3<Double, Object, Vector>, Tuple2<Object[],
                        Integer>>() {

                    private static final long serialVersionUID = -5585030372142363629L;

                    @Override
                    public void reduce(Iterable<Tuple3<Double, Object, Vector>> values,
                                       Collector<Tuple2<Object[], Integer>> out)
                            throws Exception {
                        int size = -1;
                        Set<Object> labelValues = new HashSet<>();
                        for (Tuple3<Double, Object, Vector> value : values) {
                            Tuple2<Integer, Object[]> labelVals = (Tuple2<Integer, Object[]>)value.f1;
                            for (int i = 0; i < labelVals.f1.length; ++i) {
                                labelValues.add(labelVals.f1[i]);
                            }
                            size = Math.max(size, labelVals.f0);
                        }
                        Object[] labelssort = isRegProc ? labelValues.toArray() : orderLabels(labelValues);
                        out.collect(Tuple2.of(labelssort, size));
                    }
                });
    }


    /**
     * order by the dictionary order,
     * only classification problem need do this process.
     *
     * @param unorderedLabelRows unordered label rows
     * @return sorted labels in decending lexicographical order
     */
    protected static Object[] orderLabels(Iterable<Object> unorderedLabelRows) {
        List<Object> tmpArr = new ArrayList<>();
        for (Object row : unorderedLabelRows) {
            tmpArr.add(row);
        }
        Object[] labels = tmpArr.toArray(new Object[0]);
        Preconditions.checkState((labels.length == 2), "labels count should be 2 in 2-class classification algorithm.");
        String str0 = labels[0].toString();
        String str1 = labels[1].toString();
        String positiveLabelValueString = (str1.compareTo(str0) > 0) ? str1 : str0;

        if (labels[1].toString().equals(positiveLabelValueString)) {
            Object t = labels[0];
            labels[0] = labels[1];
            labels[1] = t;
        }
        return labels;
    }

    /**
     * Do transform for train data.
     *
     * @param initData    initial data.
     * @param isRegProc   train process is regression or classification.
     * @param labelValues label values.
     * @return data for DeepFM training (weight, label, feature vector),
     * for classification label it is 0.0/1.0, for regression it is Double number.
     */
    public static DataSet<Tuple3<Double, Double, Vector>> transferLabel(
            DataSet<Tuple3<Double, Object, Vector>> initData,
            final boolean isRegProc,
            DataSet<Object[]> labelValues,
            final String LABEL_VALUES) {
        return initData.mapPartition(
                new RichMapPartitionFunction<Tuple3<Double, Object, Vector>, Tuple3<Double, Double, Vector>>() {
                    private static final long serialVersionUID = 7503654006872839366L;
                    private Object[] labelValues = null;

                    @Override
                    public void open(Configuration parameters) throws Exception {

                        this.labelValues = (Object[])getRuntimeContext()
                                .getBroadcastVariable(LABEL_VALUES).get(0);
                    }

                    @Override
                    public void mapPartition(Iterable<Tuple3<Double, Object, Vector>> values,
                                             Collector<Tuple3<Double, Double, Vector>> out) throws Exception {
                        for (Tuple3<Double, Object, Vector> value : values) {

                            if (value.f0 > 0) {
                                Double label = isRegProc ? Double.valueOf(value.f1.toString())
                                        : (value.f1.equals(labelValues[0]) ? 1.0 : 0.0);
                                out.collect(Tuple3.of(value.f0, label, value.f2));
                            }
                        }
                    }
                })
                .withBroadcastSet(labelValues, LABEL_VALUES);
    }


    /**
     * Transform train data to Tuple3 format.
     *
     * @param in     train data in row format.
     * @param params train parameters.
     * @return Tuple3 format train data <weight, label, vector></>.
     */
    public static DataSet<Tuple3<Double, Object, Vector>> transform(
            BatchOperator in,
            Params params,
            boolean isRegProc,
            String modelName) {
        TableSchema dataSchema = in.getSchema();
        String[] featureColNames;
        String labelName;
        String weightColName;
        String vectorColName;
        if (modelName.toLowerCase().equals("deepfm")) {
            featureColNames = params.get(DeepFmTrainParams.FEATURE_COLS);
            labelName = params.get(DeepFmTrainParams.LABEL_COL);
            weightColName = params.get(DeepFmTrainParams.WEIGHT_COL);
            vectorColName = params.get(DeepFmTrainParams.VECTOR_COL);
            if (null == featureColNames && null == vectorColName) {
                featureColNames = TableUtil.getNumericCols(dataSchema, new String[] {labelName});
                params.set(DeepFmTrainParams.FEATURE_COLS, featureColNames);
            }
        } else if (modelName.toLowerCase().equals("fm")) {
            featureColNames = params.get(FmTrainParams.FEATURE_COLS);
            labelName = params.get(FmTrainParams.LABEL_COL);
            weightColName = params.get(FmTrainParams.WEIGHT_COL);
            vectorColName = params.get(FmTrainParams.VECTOR_COL);
            if (null == featureColNames && null == vectorColName) {
                featureColNames = TableUtil.getNumericCols(dataSchema, new String[] {labelName});
                params.set(FmTrainParams.FEATURE_COLS, featureColNames);
            }
        } else {
            throw new NotImplementedException("This model type is not supported.");
        }

        int[] featureIndices = null;
        int labelIdx = TableUtil.findColIndexWithAssertAndHint(dataSchema.getFieldNames(), labelName);
        if (featureColNames != null) {
            featureIndices = new int[featureColNames.length];
            for (int i = 0; i < featureColNames.length; ++i) {
                int idx = TableUtil.findColIndexWithAssertAndHint(in.getColNames(), featureColNames[i]);
                featureIndices[i] = idx;
            }
        }
        int weightIdx = weightColName != null ? TableUtil.findColIndexWithAssertAndHint(in.getColNames(), weightColName)
                : -1;
        int vecIdx = vectorColName != null ? TableUtil.findColIndexWithAssertAndHint(in.getColNames(), vectorColName)
                : -1;

        return in.getDataSet().mapPartition(new Transform(isRegProc, weightIdx,
                vecIdx, featureIndices, labelIdx));
    }


    /**
     * Transform the train data to Tuple3 format: Tuple3<weightValue, labelValue, featureSparseVector>
     */
    private static class Transform extends RichMapPartitionFunction<Row, Tuple3<Double, Object, Vector>> {
        private static final long serialVersionUID = 29645096292932117L;
        private int vecIdx;
        private int labelIdx;
        private int weightIdx;
        private boolean isRegProc;
        private int[] featureIndices;

        public Transform(boolean isRegProc, int weightIndx, int vecIdx, int[] featureIndices, int labelIdx) {
            this.vecIdx = vecIdx;
            this.labelIdx = labelIdx;
            this.weightIdx = weightIndx;
            this.isRegProc = isRegProc;
            this.featureIndices = featureIndices;
        }

        @Override
        public void mapPartition(Iterable<Row> values,
                                 Collector<Tuple3<Double, Object, Vector>> out) throws Exception {
            Set<Object> labelValues = new HashSet<>();
            int size = -1;
            if (featureIndices != null) {
                size = featureIndices.length;
            }
            for (Row row : values) {
                Double weight = weightIdx == -1 ? 1.0 : ((Number)row.getField(weightIdx)).doubleValue();
                Object label = row.getField(labelIdx);

                if (!this.isRegProc) {
                    labelValues.add(label);
                } else {
                    labelValues.add(0.0);
                }

                Vector vec;
                if (featureIndices != null) {
                    vec = new DenseVector(featureIndices.length);
                    for (int i = 0; i < featureIndices.length; ++i) {
                        vec.set(i, ((Number)row.getField(featureIndices[i])).doubleValue());
                    }
                } else {
                    vec = VectorUtil.getVector(row.getField(vecIdx));
                    if (vec instanceof SparseVector) {
                        int[] indices = ((SparseVector)vec).getIndices();
                        for (int i = 0; i < indices.length; ++i) {
                            size = (vec.size() > 0) ? vec.size() : Math.max(size, indices[i] + 1);
                        }
                    } else {
                        size = ((DenseVector)vec).getData().length;
                    }
                    Preconditions.checkState((vec != null),
                            "vector for fm/deepFm model train is null, please check your input data.");
                }
                out.collect(Tuple3.of(weight, label, vec));
            }
            out.collect(
                    Tuple3.of(-1.0, Tuple2.of(size, labelValues.toArray()), new DenseVector(0)));
        }
    }
}
