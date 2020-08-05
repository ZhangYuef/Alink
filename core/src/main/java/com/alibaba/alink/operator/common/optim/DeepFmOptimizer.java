package com.alibaba.alink.operator.common.optim;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


import com.alibaba.alink.common.MLEnvironment;
import com.alibaba.alink.common.comqueue.ComContext;
import com.alibaba.alink.common.comqueue.CompareCriterionFunction;
import com.alibaba.alink.common.comqueue.CompleteResultFunction;
import com.alibaba.alink.common.comqueue.ComputeFunction;
import com.alibaba.alink.common.comqueue.IterativeComQueue;
import com.alibaba.alink.common.comqueue.communication.AllReduce;
import com.alibaba.alink.common.linalg.DenseMatrix;
import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.SparseVector;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.common.utils.JsonConverter;
import com.alibaba.alink.operator.common.classification.ann.Stacker;
import com.alibaba.alink.operator.common.classification.ann.Topology;
import com.alibaba.alink.operator.common.classification.ann.TopologyModel;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.DeepFmDataFormat;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.LogitLoss;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.LossFunction;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.SquareLoss;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.Task;
import com.alibaba.alink.operator.common.optim.subfunc.OptimVariable;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;

import com.alibaba.alink.pipeline.classification.DeepFmModel;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

/**
 * Base DeepFM model training.
 */
public class DeepFmOptimizer {
    private Params params;
    private DataSet<Tuple3<Double, Double, Vector>> trainData;
    private int[] dim;
    protected DataSet<DeepFmDataFormat> deepFmModel = null;
    private double[] lambda;
    private Topology topology;

    /**
     * construct function.
     *
     * @param trainData train data.
     * @param topology  network topology for multi-layer perceptions
     * @param params    parameters for optimizer.
     */
    public DeepFmOptimizer(DataSet<Tuple3<Double, Double, Vector>> trainData,  Params params) {
        this.params = params;
        this.trainData = trainData;

        this.dim = new int[3];
        dim[0] = params.get(DeepFmTrainParams.WITH_INTERCEPT) ? 1 : 0;
        dim[1] = params.get(DeepFmTrainParams.WITH_LINEAR_ITEM) ? 1 : 0;
        dim[2] = params.get(DeepFmTrainParams.NUM_FACTOR);

        this.lambda = new double[3];
        lambda[0] = params.get(DeepFmTrainParams.LAMBDA_0);
        lambda[1] = params.get(DeepFmTrainParams.LAMBDA_1);
        lambda[2] = params.get(DeepFmTrainParams.LAMBDA_2);
    }

    /**
     * initialize DeepFmModel.
     */
    public void setWithInitFactors(DataSet<DeepFmDataFormat> model) {
        this.deepFmModel = model;
    }

    /**
     * optimize DeepFm problem.
     *
     * @return DeepFm model.
     */
    public DataSet<DeepFmDataFormat> optimize() {
        DataSet<Row> model = new IterativeComQueue()
                .initWithPartitionedData(OptimVariable.deepFmTrainData, trainData)
                .initWithBroadcastData(OptimVariable.deepFmModel, deepFmModel)
                .add(new UpdateLocalModel(dim, lambda, params))
                .add(new AllReduce(OptimVariable.factorAllReduce))  //TODO: advise
                .add(new UpdateGlobalModel(dim))
                .add(new CalcLossAndEvaluation(dim, params.get(ModelParamName.TASK)))
                .add(new AllReduce(OptimVariable.lossAucAllReduce))  // TODO: advise
                .setCompareCriterionOfNode0(new deepFmIterTermination(params))
                .closeWith(new OutputDeepFmModel())
                .setMaxIter(Integer.MAX_VALUE)
                .exec();
        return model.mapPartition(new ParseRowModel());
    }


    /**
     * Termination class of DeepFm iteration.
     */
    public static class deepFmIterTermination extends CompareCriterionFunction {
        private static final long serialVersionUID = 1520894992934187032L;
        private double oldLoss;
        private double epsilon;
        private Task task;
        private int maxIter;
        private int batchSize;
        private Long oldTime;

        public deepFmIterTermination(Params params) {
            this.maxIter = params.get(DeepFmTrainParams.NUM_EPOCHS);
            this.epsilon = params.get(DeepFmTrainParams.EPSILON);
            this.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            this.batchSize = params.get(DeepFmTrainParams.MINIBATCH_SIZE);
            this.oldTime = System.currentTimeMillis();
        }

        @Override
        public boolean calc(ComContext context) {
            int numSamplesOnNode0
                    = ((List<Tuple3<Double, Double, Vector>>)context.getObj(OptimVariable.deepFmTrainData)).size();

            int numBatches = (batchSize == -1 || batchSize > numSamplesOnNode0) ? maxIter
                    : (numSamplesOnNode0 / batchSize + 1) * maxIter;

            double[] loss = context.getObj(OptimVariable.lossAucAllReduce);
            if (task.equals(Task.BINARY_CLASSIFICATION)) {
                System.out.println("step : " + context.getStepNo() + " loss : "
                        + loss[0] / loss[1] + "  auc : " + loss[2] / context.getNumTask() + " accuracy : "
                        + loss[3] / loss[1] + " time : " + (System.currentTimeMillis()
                        - oldTime));
                oldTime = System.currentTimeMillis();
            } else {
                System.out.println("step : " + context.getStepNo() + " loss : "
                        + loss[0] / loss[1] + "  mae : " + loss[2] / loss[1] + " mse : "
                        + loss[3] / loss[1] + " time : " + (System.currentTimeMillis()
                        - oldTime));
                oldTime = System.currentTimeMillis();
            }

            if (context.getStepNo() == numBatches) {
                return true;
            }

            if (Math.abs(oldLoss - loss[0] / loss[1]) / oldLoss < epsilon) {
                return true;
            } else {
                oldLoss = loss[0] / loss[1];
                return false;
            }
        }
    }


    /**
     * Calculate loss and evaluations.
     */
    public static class CalcLossAndEvaluation extends ComputeFunction {
        private static final long serialVersionUID = 6971431147725861845L;
        private int[] dim;
        private double[] y;
        private LossFunction lossFunc = null;
        private Task task;
        private Topology topology;
        private transient TopologyModel topologyModel = null;

        public CalcLossAndEvaluation(int[] dim, String task) {
            this.dim = dim;
            this.task = Task.valueOf(task.toUpperCase());
            if (task.equals(Task.REGRESSION)) {
                double minTarget = -1.0e20;
                double maxTarget = 1.0e20;
                double d = maxTarget - minTarget;
                d = Math.max(d, 1.0);
                maxTarget = maxTarget + d * 0.2;
                minTarget = minTarget - d * 0.2;
                lossFunc = new SquareLoss(maxTarget, minTarget);
            } else {
                lossFunc = new LogitLoss();
            }
        }

        @Override
        public void calc(ComContext context) {
            double[] buffer = context.getObj(OptimVariable.lossAucAllReduce);
            // prepare buffer vec for allReduce. the last element of vec is the weight Sum.
            if (buffer == null) {
                buffer = new double[4];
                context.putObj(OptimVariable.lossAucAllReduce, buffer);
            }
            List<Tuple3<Double, Double, Vector>> labledVectors = context.getObj(OptimVariable.deepFmTrainData);
            if (this.y == null) {
                this.y = new double[labledVectors.size()];
            }

            DenseVector coefVector = context.getObj(OptimVariable.coef);
            if (topologyModel == null) {
                topologyModel = topology.getModel(coefVector);
            } else {
                topologyModel.resetModel(coefVector);
            }

            // get DeepFmModel from static memory.
            DeepFmDataFormat factors = ((List<DeepFmDataFormat>)context.getObj(OptimVariable.deepFmModel)).get(0);
            Arrays.fill(y, 0.0);
            for (int s = 0; s < labledVectors.size(); s++) {
                Vector sample = labledVectors.get(s).f2;
                y[s] = fmCalcY(sample, factors, dim).f0;
                Double tes = topologyModel.predict((DenseVector) sample).getData()[0];
            }
            double lossSum = 0.;
            for (int i = 0; i < y.length; i++) {
                double yTruth = labledVectors.get(i).f1;
                double l = lossFunc.l(yTruth, y[i]);
                lossSum += l;
            }

            if (this.task.equals(Task.REGRESSION)) {
                double mae = 0.0;
                double mse = 0.0;
                for (int i = 0; i < y.length; i++) {
                    double yDiff = y[i] - labledVectors.get(i).f1;
                    mae += Math.abs(yDiff);
                    mse += yDiff * yDiff;
                }
                buffer[2] = mae;
                buffer[3] = mse;
            } else {
                Integer[] order = new Integer[y.length];
                double correctNum = 0.0;
                for (int i = 0; i < y.length; i++) {
                    order[i] = i;
                    if (y[i] > 0 && labledVectors.get(i).f1 > 0.5) {
                        correctNum += 1.0;
                    }
                    if (y[i] < 0 && labledVectors.get(i).f1 < 0.5) {
                        correctNum += 1.0;
                    }
                }
                Arrays.sort(order, new java.util.Comparator<Integer>() {
                    @Override
                    public int compare(Integer o1, Integer o2) {
                        return Double.compare(y[o1], y[o2]);
                    }
                });

                // mSum: positive sample number
                // nSum: negative sample number
                int mSum = 0;
                int nSum = 0;
                double posRankSum = 0.;
                for (int i = 0; i < order.length; i++) {
                    int sampleId = order[i];
                    int rank = i + 1;
                    boolean isPositiveSample = labledVectors.get(sampleId).f1 > 0.5;
                    if (isPositiveSample) {
                        mSum++;
                        posRankSum += rank;
                    } else {
                        nSum++;
                    }
                }
                if (mSum != 0 && nSum != 0) {
                    double auc = (posRankSum - 0.5 * mSum * (mSum + 1.0)) / ((double)mSum * (double)nSum);
                    buffer[2] = auc;
                    buffer[3] = correctNum;
                } else {
                    buffer[2] = 0.0;
                    buffer[3] = correctNum;
                }
            }
            buffer[0] = lossSum;
            buffer[1] = y.length;
        }
    }


    /**
     * Update global DeepFm model.
     */
    public static class UpdateGlobalModel extends ComputeFunction {
        private static final long serialVersionUID = -1476090921825562804L;
        private int[] dim;

        public UpdateGlobalModel(int[] dim) {
            this.dim = dim;
        }

        @Override
        public void calc(ComContext context) {
            double[] buffer = context.getObj(OptimVariable.factorAllReduce);
            DeepFmDataFormat sigmaGii = context.getObj(OptimVariable.sigmaGii);

            DeepFmDataFormat factors = ((List<DeepFmDataFormat>)context.getObj(OptimVariable.deepFmModel)).get(0);

            // TODO: why calc like this?
            int vectorSize = (buffer.length - 2 * dim[0]) / (2 * dim[2] + 2 * dim[1] + 1);
            int jLen = dim[2] + dim[1];
            for (int i = 0; i < vectorSize; ++i) {
                double weightSum = buffer[2 * vectorSize * jLen + i];
                if (weightSum > 0.0) {
                    for (int j = 0; j < dim[2]; ++j) {
                        factors.factors[i][j] = buffer[i * dim[2] + j] / weightSum;
                        sigmaGii.factors[i][j] = buffer[(vectorSize + i) * dim[2] + j] / weightSum;
                    }
                    if (dim[1] > 0) {
                        factors.linearItems[i] = buffer[vectorSize * dim[2] * 2 + i] / weightSum;
                        sigmaGii.linearItems[i] = buffer[vectorSize * (dim[2] * 2 + 1) + i] / weightSum;
                    }
                }
            }
            if (dim[0] > 0) {
                factors.bias = buffer[buffer.length - 2] / context.getNumTask();
                sigmaGii.bias = buffer[buffer.length - 1] / context.getNumTask();
            }
        }
    }


    /**
     * Update local DeepFm model.
     */
    public static class UpdateLocalModel extends ComputeFunction {
        private static final long serialVersionUID = -7361311383962602144L;

        /**
         * object function class, it supply the functions to calc local gradient (or loss).
         */
        private int[] dim;
        private double[] lambda;
        private Task task;
        private double learnRate;
        private int vectorSize;
        private final double eps = 1.0e-8;
        private int batchSize;
        private LossFunction lossFunc = null;
        private Random rand = new Random(2020);

        private Stacker stacker;

        public UpdateLocalModel(int[] dim, double[] lambda, Params params) {
            this.lambda = lambda;
            this.dim = dim;
            this.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            this.learnRate = params.get(DeepFmTrainParams.LEARN_RATE);
            this.batchSize = params.get(DeepFmTrainParams.MINIBATCH_SIZE);

            int[] layerSize = params.get(DeepFmTrainParams.LAYERS);
            this.stacker = new Stacker(layerSize[0], layerSize[layerSize.length - 1], true);

            if (task.equals(Task.REGRESSION)) {
                double minTarget = -1.0e20;
                double maxTarget = 1.0e20;
                double d = maxTarget - minTarget;
                d = Math.max(d, 1.0);
                maxTarget = maxTarget + d * 0.2;
                minTarget = minTarget - d * 0.2;
                lossFunc = new SquareLoss(maxTarget, minTarget);
            } else {
                lossFunc = new LogitLoss();
            }
        }

        @Override
        public void calc(ComContext context) {
            ArrayList<Tuple3<Double, Double, Vector>> labledVectors = context.getObj(OptimVariable.deepFmTrainData);

            if (batchSize == -1) {
                batchSize = labledVectors.size();
            }

            // get DeepFmModel from static memory.
            DeepFmDataFormat sigmaGii = context.getObj(OptimVariable.sigmaGii);
            DeepFmDataFormat innerModel = ((List<DeepFmDataFormat>)context.getObj(OptimVariable.deepFmModel)).get(0);
            double[] weights = context.getObj(OptimVariable.weights);

            if (weights == null) {
                vectorSize = (innerModel.factors != null) ? innerModel.factors.length : innerModel.linearItems.length;
                weights = new double[vectorSize];
                context.putObj(OptimVariable.weights, weights);
            } else {
                Arrays.fill(weights, 0.);
            }

            if (sigmaGii == null) {
                sigmaGii = new DeepFmDataFormat(vectorSize, dim, 0.0);
                context.putObj(OptimVariable.sigmaGii, sigmaGii);
            }

            updateFactors(labledVectors, innerModel, learnRate, sigmaGii, weights);

            // dim[0] - WITH_INTERCEPT, dim[1] - WITH_LINEAR_ITEM, dim[2] - NUM_FACTOR
            // prepare buffer vec for allReduce. the last element of vec is the weight Sum.
            double[] buffer = context.getObj(OptimVariable.factorAllReduce);
            if (buffer == null) {
                buffer = new double[vectorSize * (dim[1] + dim[2]) * 2 + vectorSize + 2 * dim[0]];
                context.putObj(OptimVariable.factorAllReduce, buffer);
            } else {
                Arrays.fill(buffer, 0.0);
            }

            for (int i = 0; i < vectorSize; ++i) {
                for (int j = 0; j < dim[2]; ++j) {
                    buffer[i * dim[2] + j] = innerModel.factors[i][j] * weights[i];
                    buffer[(vectorSize + i) * dim[2] + j] = sigmaGii.factors[i][j] * weights[i];
                }
                if (dim[1] > 0) {
                    buffer[vectorSize * dim[2] * 2 + i] = innerModel.linearItems[i] * weights[i];
                    buffer[vectorSize * (dim[2] * 2 + dim[1]) + i] = sigmaGii.linearItems[i] * weights[i];
                }
                buffer[vectorSize * ((dim[2] + dim[1]) * 2) + i] = weights[i];
            }
            if (dim[0] > 0) {
                buffer[vectorSize * ((dim[2] + dim[1]) * 2 + 1)] = innerModel.bias;
                buffer[vectorSize * ((dim[2] + dim[1]) * 2 + 1) + 1] = sigmaGii.bias;
            }
        }

        private void updateFactors(List<Tuple3<Double, Double, Vector>> labledVectors,
                                   DeepFmDataFormat factors,
                                   double learnRate,
                                   DeepFmDataFormat sigmaGii,
                                   double[] weights) {
            TopologyModel topologyModel = factors.topologyModel;
            for (int bi = 0; bi < batchSize; ++bi) {
                Tuple3<Double, Double, Vector> sample = labledVectors.get(rand.nextInt(labledVectors.size()));
                Vector vec = sample.f2;
                //TODO: MLP forward
                Tuple2<Double, double[]> yVx = fmCalcY(vec, factors, dim);   // y, vx[]
                // TODO: calc Y by MLP, how to update FM and MLP together
                // adaptive data type
//                Tuple2<DenseMatrix, DenseMatrix> unstacked = stacker.unstack(sample);
//                topologyModel.computeGradient(unstacked.f0, unstacked.f1, updateGrad);
                DenseVector yDeep = topologyModel.predict((DenseVector) vec);

                double yTruth = sample.f1;
                double dldy = lossFunc.dldy(yTruth, yVx.f0);

                int[] indices;
                double[] vals;
                if (sample.f2 instanceof SparseVector) {
                    indices = ((SparseVector)sample.f2).getIndices();
                    vals = ((SparseVector)sample.f2).getValues();
                } else {
                    indices = new int[sample.f2.size()];
                    for (int i = 0; i < sample.f2.size(); ++i) {
                        indices[i] = i;
                    }
                    vals = ((DenseVector)sample.f2).getData();
                }

                // TODO: how to calc here?
                if (dim[0] > 0) {
                    double grad = dldy + lambda[0] * factors.bias;

                    sigmaGii.bias += grad * grad;
                    factors.bias += -learnRate * grad / (Math.sqrt(sigmaGii.bias + eps));
                }

                for (int i = 0; i < indices.length; ++i) {
                    int idx = indices[i];

                    weights[idx] += sample.f0;
                    // update DeepFmModel
                    // TODO: how to calc?
                    for (int j = 0; j < dim[2]; j++) {
                        double vixi = vals[i] * factors.factors[idx][j];
                        double d = vals[i] * (yVx.f1[j] - vixi);
                        double grad = dldy * d + lambda[2] * factors.factors[idx][j];
                        sigmaGii.factors[idx][j] += grad * grad;
                        factors.factors[idx][j] += -learnRate * grad / (Math.sqrt(sigmaGii.factors[idx][j] + eps));
                    }
                    if (dim[1] > 0) {
                        double grad = dldy * vals[i] + lambda[1] * factors.linearItems[idx];
                        sigmaGii.linearItems[idx] += grad * grad;
                        factors.linearItems[idx] += -grad * learnRate / (Math.sqrt(sigmaGii.linearItems[idx] + eps));
                    }
                }
            }
        }
    }


    /**
     * Output DeepFm model with row format.
     */
    public static class OutputDeepFmModel extends CompleteResultFunction {
        private static final long serialVersionUID = -1918870713589656276L;

        @Override
        public List<Row> calc(ComContext context) {
            if (context.getTaskId() != 0) {
                return null;
            }
            DeepFmDataFormat factors = ((List<DeepFmDataFormat>)context.getObj(OptimVariable.deepFmModel)).get(0);
            List<Row> model = new ArrayList<>();
            model.add(Row.of(JsonConverter.toJson(factors)));
            return model;
        }
    }


    public class ParseRowModel extends RichMapPartitionFunction<Row, DeepFmDataFormat> {
        private static final long serialVersionUID = 2674679974431661448L;

        @Override
        public void mapPartition(Iterable<Row> iterable,
                                 Collector<DeepFmDataFormat> collector) throws Exception {
            int taskId = getRuntimeContext().getIndexOfThisSubtask();
            if (taskId == 0) {
                for (Row row : iterable) {
                    DeepFmDataFormat factor = JsonConverter.fromJson((String)row.getField(0), DeepFmDataFormat.class);
                    collector.collect(factor);
                }
            }
        }
    }

    /**
     * calculate the value of y with given fm model.
     *
     * @param vec           input data vector
     * @param deepFmModel
     * @param dim
     * @return
     */
    public static Tuple2<Double, double[]> fmCalcY(Vector vec, DeepFmDataFormat deepFmModel, int[] dim) {
        int[] featureIds;
        double[] featureValues;
        if (vec instanceof SparseVector) {
            featureIds = ((SparseVector)vec).getIndices();
            featureValues = ((SparseVector)vec).getValues();
        } else {
            featureIds = new int[vec.size()];
            for (int i = 0; i < vec.size(); ++i) {
                featureIds[i] = i;
            }
            featureValues = ((DenseVector)vec).getData();
        }

        double[] vx = new double[dim[2]];
        double[] v2x2 = new double[dim[2]];

        // (1) compute y
        double y = 0.;

        if (dim[0] > 0) {
            y += deepFmModel.bias;
        }

        for (int i = 0; i < featureIds.length; i++) {
            int featurePos = featureIds[i];
            double x = featureValues[i];

            // the linear term
            if (dim[1] > 0) {
                y += x * deepFmModel.linearItems[featurePos];
            }
            // the quadratic term
            for (int j = 0; j < dim[2]; j++) {
                double vixi = x * deepFmModel.factors[featurePos][j];
                vx[j] += vixi;
                v2x2[j] += vixi * vixi;
            }
        }

        for (int i = 0; i < dim[2]; i++) {
            y += 0.5 * (vx[i] * vx[i] - v2x2[i]);
        }
        return Tuple2.of(y, vx);
    }

}
