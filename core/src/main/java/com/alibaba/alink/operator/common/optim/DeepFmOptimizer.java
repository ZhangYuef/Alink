package com.alibaba.alink.operator.common.optim;

import java.util.*;

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
import com.alibaba.alink.operator.common.classification.ann.FeedForwardTopology;
import com.alibaba.alink.operator.common.classification.ann.Stacker;
import com.alibaba.alink.operator.common.classification.ann.Topology;
import com.alibaba.alink.operator.common.classification.ann.TopologyModel;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.DeepFmDataFormat;
import com.alibaba.alink.operator.common.fm.FmLossUtils;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.Task;
import com.alibaba.alink.operator.common.optim.subfunc.OptimVariable;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;
import com.alibaba.alink.operator.common.utils.FmOptimizerUtils;
import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import static com.alibaba.alink.operator.common.utils.FmOptimizerUtils.calcGradient;

/**
 * Base DeepFM model training.
 */
public class DeepFmOptimizer {
    private Params params;
    private DataSet<Tuple3<Double, Double, Vector>> trainData;
    private int[] dim;
    protected DataSet<DeepFmDataFormat> deepFmModel = null;
    private double[] lambda;

    /**
     * construct function.
     *
     * @param trainData train data.
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
                .add(new AllReduce(OptimVariable.factorAllReduce))
                .add(new UpdateGlobalModel(dim))
                .add(new CalcLossAndEvaluation(dim, params.get(ModelParamName.TASK)))
                .add(new AllReduce(OptimVariable.lossAucAllReduce))
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
        private FmLossUtils.LossFunction lossFunc = null;
        private Task task;

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
                lossFunc = new FmLossUtils.SquareLoss(maxTarget, minTarget);
            } else {
                lossFunc = new FmLossUtils.LogitLoss();
            }
        }

        @Override
        public void calc(ComContext context) {
            // buffer vector for AllReduce. buffer[0] - loss sum, buffer[1] - input size,
            // for regression task: buffer[2] - MAE, buffer[3] - MSE
            // for classification task: buffer[2] - AUC, buffer[3] - correct number count
            double[] buffer = context.getObj(OptimVariable.lossAucAllReduce);
            if (buffer == null) {
                buffer = new double[4];
                context.putObj(OptimVariable.lossAucAllReduce, buffer);
            }
            List<Tuple3<Double, Double, Vector>> labledVectors = context.getObj(OptimVariable.deepFmTrainData);
            if (this.y == null) {
                this.y = new double[labledVectors.size()];
            }

            // get DeepFmModel from static memory.
            DeepFmDataFormat factors = ((List<DeepFmDataFormat>)context.getObj(OptimVariable.deepFmModel)).get(0);
            Arrays.fill(y, 0.0);
            for (int s = 0; s < labledVectors.size(); s++) {
                Vector sample = labledVectors.get(s).f2;
                y[s] = FmOptimizerUtils.fmCalcY(sample, factors.linearItems, factors.factors, factors.bias, dim).f0
                        + deepCalcY(factors, dim).get(0);
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
                } else {
                    buffer[2] = 0.0;
                }
                buffer[3] = correctNum;
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
            Tuple2<DenseVector, double[]> grad = factors.dir;
            int size = grad.f0.size();

            // TODO: why calc like this?
            // vectorSize * (dim[1] + dim[2]) * 2 + vectorSize + 2 * dim[0] + size + 2
            int vectorSize = (buffer.length - 2 * dim[0] - size - 2) / (2 * (dim[2] + dim[1]) + 1);
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

            // deep part
            int start = vectorSize * (dim[1] + dim[2]) * 2 + vectorSize + 2 * dim[0];
            for (int i = start; i < start + size; i++) {
                grad.f0.set(i - start, buffer[i] / buffer[start + size]);
            }
            grad.f1[0] = buffer[start + size];
            grad.f1[1] = buffer[start + size + 1];
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
        private FmLossUtils.LossFunction lossFunc = null;
        private Random rand = new Random(2020);
        private Topology topology = null;
        private TopologyModel topologyModel = null;

        private Stacker stacker = null;

        public UpdateLocalModel(int[] dim, double[] lambda, Params params) {
            this.lambda = lambda;
            this.dim = dim;
            this.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            this.learnRate = params.get(DeepFmTrainParams.LEARN_RATE);
            this.batchSize = params.get(DeepFmTrainParams.MINIBATCH_SIZE);

            if (task.equals(Task.REGRESSION)) {
                double minTarget = -1.0e20;
                double maxTarget = 1.0e20;
                double d = maxTarget - minTarget;
                d = Math.max(d, 1.0);
                maxTarget = maxTarget + d * 0.2;
                minTarget = minTarget - d * 0.2;
                lossFunc = new FmLossUtils.SquareLoss(maxTarget, minTarget);
            } else {
                lossFunc = new FmLossUtils.LogitLoss();
            }
        }

        @Override
        public void calc(ComContext context) {
            ArrayList<Tuple3<Double, Double, Vector>> labledVectors = context.getObj(OptimVariable.deepFmTrainData);

            if (batchSize == -1) {
                batchSize = labledVectors.size();
            }

            // get DeepFmModel iterative state from static memory.
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

            if (stacker == null) {
                int[] layerSize = innerModel.layerSize;
                this.stacker = new Stacker(layerSize[0], layerSize[layerSize.length - 1], true);
            }

            DenseVector coefVector = innerModel.coefVector;
            int size = coefVector.size();
            // calculate local gradient
            Tuple2<DenseVector, double[]> grad = innerModel.dir;
            for (int i = 0; i < size; ++i) {
                grad.f0.set(i, 0.0);
            }

            updateFactors(labledVectors, innerModel, learnRate, sigmaGii, weights, grad);

            // dim[0] - WITH_INTERCEPT, dim[1] - WITH_LINEAR_ITEM, dim[2] - NUM_FACTOR
            // prepare buffer vec for allReduce. the last element of vec is the weight Sum.
            double[] buffer = context.getObj(OptimVariable.factorAllReduce);
            if (buffer == null) {
                buffer = new double[vectorSize * (dim[1] + dim[2]) * 2 + vectorSize + 2 * dim[0] + size + 2];
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

            // deep part
            int start = vectorSize * (dim[1] + dim[2]) * 2 + vectorSize + 2 * dim[0];
            for (int i = start; i < start + size; i++) {
                buffer[i] = grad.f0.get(i - start) * grad.f1[0];
            }
            buffer[size] = grad.f1[0];
            buffer[size + 1] = grad.f1[1];
        }

        private void updateFactors(List<Tuple3<Double, Double, Vector>> labledVectors,
                                   DeepFmDataFormat factors,
                                   double learnRate,
                                   DeepFmDataFormat sigmaGii,
                                   double[] weights,
                                   Tuple2<DenseVector, double[]> grads) {
            if (topology == null) {
                topology = FeedForwardTopology.multiLayerPerceptron(factors.layerSize, false, factors.dropoutRate);
            }
            if (topologyModel == null) {
                topologyModel = topology.getModel(factors.coefVector);
            } else {
                topologyModel.resetModel(factors.coefVector);
            }
            double weightSum = 0.0;
            double loss = 0.0;

            DenseVector coefVector = factors.coefVector;
            for (int bi = 0; bi < batchSize; ++bi) {
                Tuple3<Double, Double, Vector> sample = labledVectors.get(rand.nextInt(labledVectors.size()));
                Vector vec = sample.f2;
                Tuple2<Double, double[]> yVx = FmOptimizerUtils.fmCalcY(vec, factors.linearItems,
                        factors.factors, factors.bias, dim);
                DenseVector yDeep = deepCalcY(factors, dim);
                Double y = yVx.f0 + yDeep.get(0);

                double yTruth = sample.f1;
                double dldy = lossFunc.dldy(yTruth, y);

                // update fm part params
                Tuple3<Tuple3<Double, double[], double[][]>,
                        Tuple3<Double, double[], double[][]>,
                        double[]> result
                        = calcGradient(
                        Tuple3.of(sigmaGii.bias, sigmaGii.linearItems, sigmaGii.factors),
                        Tuple3.of(factors.bias, factors.linearItems, factors.factors),
                        yVx,
                        sample,
                        weights,
                        dldy,
                        lambda,
                        dim,
                        learnRate,
                        eps);

                sigmaGii.bias = result.f0.f0;
                sigmaGii.linearItems = result.f0.f1;
                sigmaGii.factors = result.f0.f2;
                factors.bias = result.f1.f0;
                factors.linearItems = result.f1.f1;
                factors.factors = result.f1.f2;
                weights = result.f2;

                // deep part
                for (int i = 0; i < grads.f0.size(); i++) {
                    grads.f0.set(i, 0.0);
                }
                if (sample.f2 instanceof SparseVector) {
                    ((SparseVector)(sample.f2)).setSize(coefVector.size());
                }
                weightSum += sample.f0;

                // calculate local gradient
                topologyModel.resetModel(coefVector);
                Tuple2<DenseMatrix, DenseMatrix> unstacked = stacker.unstack(sample);
                topologyModel.computeGradient(unstacked.f0, unstacked.f1, grads.f0);
                // object value
                topologyModel.resetModel(coefVector);
                double tmpLoss = topologyModel.computeGradient(unstacked.f0, unstacked.f1, null);

                loss += tmpLoss * sample.f0;
            }

            if (weightSum != 0.0) {
                grads.f0.scaleEqual(1.0 / weightSum);
                loss /= weightSum;
            }
            grads.f1[0] = weightSum;
            grads.f1[1] = loss;
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



    public static DenseVector deepCalcY(DeepFmDataFormat deepFmModel, int[] dim) {
        // convert and concatenate embeddings into 1 dimension
        int vecSize = deepFmModel.factors.length;
        DenseVector input = new DenseVector(vecSize * dim[2]);
        for (int i = 0; i < vecSize; i++) {
            for (int j = 0; j < dim[2]; j++) {
                input.set(i * dim[2] + j, deepFmModel.factors[i][j]);
            }
        }

        Topology topology = FeedForwardTopology.multiLayerPerceptron(deepFmModel.layerSize, false, deepFmModel.dropoutRate);

        TopologyModel topologyModel = topology.getModel(deepFmModel.coefVector);

        DenseVector output = topologyModel.predict(input);

        return output;
    }

}
