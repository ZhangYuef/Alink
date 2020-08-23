package com.alibaba.alink.operator.common.deepfm;

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
import com.alibaba.alink.operator.common.classification.ann.FeedForwardTopology;
import com.alibaba.alink.operator.common.classification.ann.Topology;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;
import com.alibaba.alink.operator.common.fm.FmUtils;

import org.apache.flink.api.common.functions.*;
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
 * Base DeepFM model training.
 */
public abstract class BaseDeepFmTrainBatchOp<T extends BaseDeepFmTrainBatchOp<T>> extends BatchOperator<T> implements
        WithModelInfoBatchOp<DeepFmModelInfo, T, DeepFmModelInfoBatchOp> {

    public static final String LABEL_VALUES = "labelValues";
    public static final String VEC_SIZE = "vecSize";
    private static final long serialVersionUID = 1862222879134609596L;
    private int[] dim;
    private TypeInformation labelType;

    /**
     * @param params parameters needed by training process.
     */
    public BaseDeepFmTrainBatchOp(Params params) {
        super(params);
    }

    /**
     * construct function.
     */
    public BaseDeepFmTrainBatchOp() {
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
     * @return model.
     */
    protected abstract DataSet<BaseDeepFmTrainBatchOp.DeepFmDataFormat> optimize(DataSet<Tuple3<Double, Double, Vector>> trainData,
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

        // FM part parameters.
        Params params = getParams();

        this.dim = new int[3];
        dim[0] = params.get(DeepFmTrainParams.WITH_INTERCEPT) ? 1 : 0;
        dim[1] = params.get(DeepFmTrainParams.WITH_LINEAR_ITEM) ? 1 : 0;
        dim[2] = params.get(DeepFmTrainParams.NUM_FACTOR);

        boolean isRegProc = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase()).equals(Task.REGRESSION);
        this.labelType = isRegProc ? Types.DOUBLE : in.getColTypes()[TableUtil
                .findColIndex(in.getColNames(), params.get(DeepFmTrainParams.LABEL_COL))];

        // Deep part parameters.
        final int blockSize = params.get(DeepFmTrainParams.BLOCK_SIZE);

        // Transform data to Tuple3 format <weight, label, feature vector>.
        DataSet<Tuple3<Double, Object, Vector>> initData = FmUtils.transform(in, params, isRegProc, "DeepFm");

        // Get some util info, (label values, feature size)
        DataSet<Tuple2<Object[], Integer>> utilInfo = FmUtils.getUtilInfo(initData, isRegProc);
        DataSet<Integer> featSize = utilInfo.map(
                new MapFunction<Tuple2<Object[], Integer>, Integer>() {
                    private static final long serialVersionUID = 5387308178360941949L;

                    @Override
                    public Integer map(Tuple2<Object[], Integer> value)
                            throws Exception {
                        return value.f1;
                    }
                });
        DataSet<Object[]> labelValues = utilInfo.flatMap(
                new FlatMapFunction<Tuple2<Object[], Integer>, Object[]>() {
                    private static final long serialVersionUID = 6650886382660914850L;

                    @Override
                    public void flatMap(Tuple2<Object[], Integer> value,
                                        Collector<Object[]> out)
                            throws Exception {
                        out.collect(value.f0);
                    }
                });
        // Transform data to format: <weight, label, features>
        DataSet<Tuple3<Double, Double, Vector>> trainData = FmUtils.transferLabel(initData, isRegProc, labelValues, LABEL_VALUES);

        // Solve the optimization problem.
        DataSet<DeepFmDataFormat> model = optimize(trainData, featSize, params, dim,
                                                   MLEnvironmentFactory.get(getMLEnvironmentId()));

        // Build model rows which is the output format.
        DataSet<Row> modelRows = model.flatMap(new GenerateModelRows(params, dim, labelType, isRegProc))
                                      .withBroadcastSet(labelValues, LABEL_VALUES)
                                      .withBroadcastSet(featSize, VEC_SIZE);
        // Convert the model rows to table.
        this.setOutput(modelRows, new DeepFmModelDataConverter(labelType).getModelSchema());
        return (T)this;
    }


    /**
     * generate model in row format.
     */
    public static class GenerateModelRows extends RichFlatMapFunction<DeepFmDataFormat, Row> {

        private static final long serialVersionUID = -719029590703734326L;
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
        public void flatMap(DeepFmDataFormat value, Collector<Row> out) throws Exception {
            DeepFmModelData modelData = new DeepFmModelData();
            modelData.deepFmModel = value;
            modelData.vectorColName = params.get(DeepFmTrainParams.VECTOR_COL);
            modelData.featureColNames = params.get(DeepFmTrainParams.FEATURE_COLS);
            modelData.dim = dim;
            modelData.labelColName = params.get(DeepFmTrainParams.LABEL_COL);
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
            new DeepFmModelDataConverter(labelType).save(modelData, out);
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
     * the data structure of DeepFM model data.
     */
    public static class DeepFmDataFormat implements Serializable {
        public double[] linearItems;
        public double[][] factors;
        public double bias;
        public int[] dim;
        public int[] layerSize;
        public DenseVector initialWeights;
        public Tuple2<DenseVector, double[]> dir = null;
        public double dropoutRate;

        // empty constructor to make it POJO
        public DeepFmDataFormat() {
        }

        /**
         *
         * @param vecSize   input feature's max dimension
         * @param dim       dim[0]-with interception, dim[1]-with linear item, dim[2]-factor number
         * @param initStdev initial standard deviation for Gausssain distribution.
         */
        public DeepFmDataFormat(int vecSize, int[] dim, double initStdev) {
            this.dim = dim;
            if (dim[1] > 0) {
                this.linearItems = new double[vecSize];
            }
            if (dim[2] > 0) {
                this.factors = new double[vecSize][dim[2]];
            }
            reset(initStdev);
        }

        /**
         *
         * @param vecSize         input feature's max dimension
         * @param dim             dim[0]-with interception, dim[1]-with linear item, dim[2]-factor number
         * @param layerSize       each layers' size in MLP
         * @param initialWeights  initial weights for MLP
         * @param dropoutRate     dropout rate for MLP's dropout layer
         * @param initStdev       initial standard deviation for Gausssain distribution.
         */
        public DeepFmDataFormat(int vecSize, int[] dim, int[] layerSize, DenseVector initialWeights, double dropoutRate, double initStdev) {
            this.dim = dim;
            if (dim[1] > 0) {
                this.linearItems = new double[vecSize];
            }
            if (dim[2] > 0) {
                this.factors = new double[vecSize][dim[2]];
            }
            this.initialWeights = initialWeights;
            this.dropoutRate = dropoutRate;

            // insert vectorSize * factorSize as the first layer's input size, 1 as the final layer's output size
            int inputSize = vecSize * dim[2];
            int[] layerSizeInsert = new int[layerSize.length + 2];
            layerSizeInsert[0] = inputSize;
            for (int i = 0; i < layerSize.length; i++) {
                layerSizeInsert[i + 1] = layerSize[i];
            }
            layerSizeInsert[layerSizeInsert.length - 1] = 1;
            this.layerSize = layerSizeInsert;

            reset(this.layerSize, initStdev);
        }

        /**
         *
         * @param vecSize         input feature's max dimension
         * @param numField        field number
         * @param dim             dim[0]-with interception, dim[1]-with linear item, dim[2]-factor number
         * @param initStdev       initial standard deviation for Gausssain distribution.
         */
        public DeepFmDataFormat(int vecSize, int numField, int[] dim, double initStdev) {
            this.dim = dim;
            if (dim[1] > 0) {
                this.linearItems = new double[vecSize];
            }
            if (dim[2] > 0) {
                this.factors = new double[vecSize * numField][dim[2]];
            }
            reset(initStdev);
        }

        /**
         * Reset weight vectors for deepFM not including deep (MLP) part.
         * @param initStdev initial standard deviation for Gaussian distribution.
         */
        public void reset(double initStdev) {
            Random rand = new Random(2020);

            // FM part
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

        /**
         * Reset weight vectors for deepFM including deep (MLP) part.
         * @param layerSize       each layers' size in MLP
         * @param initStdev initial standard deviation for Gaussian distribution.
         */
        public void reset(int[] layerSize, double initStdev) {
            Random rand = new Random(2020);

            // FM part
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

            // deep part
            Topology topology = FeedForwardTopology.multiLayerPerceptron(layerSize, false, dropoutRate);
            DenseVector coefVector;
            if (initialWeights != null) {
                if (initialWeights.size() != topology.getWeightSize()) {
                    throw new RuntimeException("Invalid initial weights, size mismatch");
                }
                coefVector = initialWeights;
            } else {
                DenseVector weights = DenseVector.zeros(topology.getWeightSize());
                for (int i = 0; i < weights.size(); i++) {
                    weights.set(i, rand.nextGaussian() * initStdev);
                }
                coefVector = weights;
            }

            DenseVector vec = new DenseVector(coefVector.size());
            dir = Tuple2.of(vec, new double[2]);
            dir.f0 = coefVector;
        }
    }

    /**
     * get model info of this train process.
     *
     * @return
     */
    @Override
    public DeepFmModelInfoBatchOp getModelInfoBatchOp() {
        return new DeepFmModelInfoBatchOp(this.labelType).linkFrom(this);
    }
}