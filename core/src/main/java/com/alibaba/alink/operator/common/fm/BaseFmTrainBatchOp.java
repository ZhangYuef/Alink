package com.alibaba.alink.operator.common.fm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import com.alibaba.alink.common.MLEnvironment;
import com.alibaba.alink.common.MLEnvironmentFactory;
import com.alibaba.alink.common.lazy.WithModelInfoBatchOp;
import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.SparseVector;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.linalg.VectorUtil;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.common.utils.TableUtil;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.params.recommendation.FmTrainParams;
import com.alibaba.alink.operator.common.fm.FmUtils;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.table.api.TableSchema;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;
import org.apache.flink.util.Preconditions;

/**
 * Base FM model training.
 */
public abstract class BaseFmTrainBatchOp<T extends BaseFmTrainBatchOp<T>> extends BatchOperator<T> implements
    WithModelInfoBatchOp<FmModelInfo, T, FmModelInfoBatchOp> {

    public static final String LABEL_VALUES = "labelValues";
    public static final String VEC_SIZE = "vecSize";
    private static final long serialVersionUID = -5308557491809175331L;
    private int[] dim;
    private TypeInformation labelType;

    /**
     * @param params parameters needed by training process.
     */
    public BaseFmTrainBatchOp(Params params) {
        super(params);
    }

    /**
     * construct function.
     */
    public BaseFmTrainBatchOp() {
        super(new Params());
    }

    /**
     * The api for optimizer.
     *
     * @param trainData training Data.
     * @param vecSize   vector size.
     * @param params    parameters.
     * @param dim       dimension.
     * @param session   environment.
     * @return model coefficient.
     */
    protected abstract DataSet<FmDataFormat> optimize(DataSet<Tuple3<Double, Double, Vector>> trainData,
                                                      DataSet<Integer> vecSize,
                                                      final Params params,
                                                      final int[] dim,
                                                      MLEnvironment session);

    /**
     * do the operation of this op.
     *
     * @param inputs the linked inputs
     * @return this class.
     */
    @Override
    public T linkFrom(BatchOperator<?>... inputs) {
        BatchOperator<?> in = checkAndGetFirst(inputs);
        // Get parameters of this algorithm.
        Params params = getParams();

        this.dim = new int[3];
        dim[0] = params.get(FmTrainParams.WITH_INTERCEPT) ? 1 : 0;
        dim[1] = params.get(FmTrainParams.WITH_LINEAR_ITEM) ? 1 : 0;
        dim[2] = params.get(FmTrainParams.NUM_FACTOR);

        boolean isRegProc = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase()).equals(Task.REGRESSION);
        this.labelType = isRegProc ? Types.DOUBLE : in.getColTypes()[TableUtil
            .findColIndex(in.getColNames(), params.get(FmTrainParams.LABEL_COL))];

        // Transform data to Tuple3 format <weight, label, feature vector>.
        DataSet<Tuple3<Double, Object, Vector>> initData = FmUtils.transform(in, params, isRegProc, "Fm");

        // Get some util info, such as featureSize and labelValues.
        DataSet<Tuple2<Object[], Integer>> utilInfo = FmUtils.getUtilInfo(initData, isRegProc);
        DataSet<Integer> featSize = utilInfo.map(
            new MapFunction<Tuple2<Object[], Integer>, Integer>() {
                private static final long serialVersionUID = 1099531852518545431L;

                @Override
                public Integer map(Tuple2<Object[], Integer> value)
                    throws Exception {
                    return value.f1;
                }
            });
        DataSet<Object[]> labelValues = utilInfo.flatMap(
            new FlatMapFunction<Tuple2<Object[], Integer>, Object[]>() {
                private static final long serialVersionUID = -4407775357759305675L;

                @Override
                public void flatMap(Tuple2<Object[], Integer> value,
                                    Collector<Object[]> out)
                    throws Exception {
                    out.collect(value.f0);
                }
            });

        DataSet<Tuple3<Double, Double, Vector>>
            trainData = FmUtils.transferLabel(initData, isRegProc, labelValues, LABEL_VALUES);

        DataSet<FmDataFormat> model
            = optimize(trainData, featSize, params, dim, MLEnvironmentFactory.get(getMLEnvironmentId()));

        DataSet<Row> modelRows = model.flatMap(new GenerateModelRows(params, dim, labelType, isRegProc))
            .withBroadcastSet(labelValues, LABEL_VALUES)
            .withBroadcastSet(featSize, VEC_SIZE);

        this.setOutput(modelRows, new FmModelDataConverter(labelType).getModelSchema());
        return (T)this;
    }

    /**
     * generate model in row format.
     */
    public static class GenerateModelRows extends RichFlatMapFunction<FmDataFormat, Row> {
        private static final long serialVersionUID = -380930181466110905L;
        private Params params;
        private int[] dim;
        private TypeInformation labelType;
        private Object[] labelValues;
        private boolean isRegProc;
        private int[] fieldPos;
        private int vecSize;

        public GenerateModelRows(Params params, int[] dim, TypeInformation labelType, boolean isRegProc) {
            this.params = params;
            this.labelType = labelType;
            this.dim = dim;
            this.isRegProc = isRegProc;
        }

        @Override
        public void open(Configuration parameters) throws Exception {
            super.open(parameters);
            this.labelValues = (Object[])getRuntimeContext().getBroadcastVariable(LABEL_VALUES).get(0);
            this.vecSize = (int)getRuntimeContext().getBroadcastVariable(VEC_SIZE).get(0);
        }

        @Override
        public void flatMap(FmDataFormat value, Collector<Row> out) throws Exception {
            FmModelData modelData = new FmModelData();
            modelData.fmModel = value;
            modelData.vectorColName = params.get(FmTrainParams.VECTOR_COL);
            modelData.featureColNames = params.get(FmTrainParams.FEATURE_COLS);
            modelData.dim = dim;
            modelData.labelColName = params.get(FmTrainParams.LABEL_COL);
            modelData.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            if (fieldPos != null) {
                modelData.fieldPos = fieldPos;
            }
            if (!isRegProc) {
                modelData.labelValues = this.labelValues;
            } else {
                modelData.labelValues = new Object[] {0.0};
            }

            modelData.vectorSize = vecSize;
            new FmModelDataConverter(labelType).save(modelData, out);
        }
    }

    public enum Task {
        /**
         * regression problem.
         */
        REGRESSION,
        /**
         * binary classification problem.
         */
        BINARY_CLASSIFICATION
    }

    /**
     * loss function interface
     */
    public interface LossFunction extends Serializable {
        /**
         * calculate loss of sample.
         *
         * @param yTruth
         * @param y
         * @return
         */
        double l(double yTruth, double y);

        /**
         * calculate dldy of sample
         *
         * @param yTruth
         * @param y
         * @return
         */
        double dldy(double yTruth, double y);
    }

    /**
     * loss function for regression task
     */
    public static class SquareLoss implements LossFunction {
        private static final long serialVersionUID = -3903508209287601504L;
        private double maxTarget;
        private double minTarget;

        public SquareLoss(double maxTarget, double minTarget) {
            this.maxTarget = maxTarget;
            this.minTarget = minTarget;
        }

        @Override
        public double l(double yTruth, double y) {
            return (yTruth - y) * (yTruth - y);
        }

        @Override
        public double dldy(double yTruth, double y) {
            // a trick borrowed from libFM
            y = Math.min(y, maxTarget);
            y = Math.max(y, minTarget);

            return 2.0 * (y - yTruth);
        }
    }

    /**
     * loss function for binary classification task
     */
    public static class LogitLoss implements LossFunction {
        private static final long serialVersionUID = -166213844104644622L;

        @Override
        public double l(double yTruth, double y) { // yTruth in {0, 1}
            double logit = sigmoid(y);
            if (yTruth < 0.5) {
                return -Math.log(1. - logit);
            } else if (yTruth > 0.5) {
                return -Math.log(logit);
            } else {
                throw new RuntimeException("Invalid label: " + yTruth);
            }
        }

        @Override
        public double dldy(double yTruth, double y) {
            return sigmoid(y) - yTruth;
        }

        private double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }
    }

    /**
     * the data structure of FM model data.
     */
    public static class FmDataFormat implements Serializable {
        public double[] linearItems;
        public double[][] factors;
        public double bias;
        public int[] dim;

        // empty constructor to make it POJO
        public FmDataFormat() {
        }

        public FmDataFormat(int vecSize, int[] dim, double initStdev) {
            this.dim = dim;
            if (dim[1] > 0) {
                this.linearItems = new double[vecSize];
            }
            if (dim[2] > 0) {
                this.factors = new double[vecSize][dim[2]];
            }
            reset(initStdev);
        }

        public FmDataFormat(int vecSize, int numField, int[] dim, double initStdev) {
            this.dim = dim;
            if (dim[1] > 0) {
                this.linearItems = new double[vecSize];
            }
            if (dim[2] > 0) {
                this.factors = new double[vecSize * numField][dim[2]];
            }
            reset(initStdev);
        }

        public void reset(double initStdev) {
            Random rand = new Random(2020);
            if (dim[1] > 0) {
                for (int i = 0; i < linearItems.length; ++i) {
                    linearItems[i] = rand.nextGaussian() * initStdev;
                }
            }
            if (dim[2] > 0) {
                for (int i = 0; i < factors.length; ++i) {
                    for (int j = 0; j < dim[2]; ++j) {
                        factors[i][j] = rand.nextGaussian() * initStdev;
                    }
                }
            }
        }

    }

    /**
     * get model info of this train process.
     *
     * @return
     */
    @Override
    public FmModelInfoBatchOp getModelInfoBatchOp() {
        return new FmModelInfoBatchOp(this.labelType).linkFrom(this);
    }
}



