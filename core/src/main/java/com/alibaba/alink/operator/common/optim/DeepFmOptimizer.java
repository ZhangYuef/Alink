package com.alibaba.alink.operator.common.optim;

import java.util.*;
import java.util.stream.IntStream;

import com.alibaba.alink.common.comqueue.ComContext;
import com.alibaba.alink.common.comqueue.CompareCriterionFunction;
import com.alibaba.alink.common.comqueue.CompleteResultFunction;
import com.alibaba.alink.common.comqueue.ComputeFunction;
import com.alibaba.alink.common.comqueue.IterativeComQueue;
import com.alibaba.alink.common.comqueue.communication.AllReduce;
import com.alibaba.alink.common.linalg.*;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.common.utils.JsonConverter;
import com.alibaba.alink.operator.common.classification.ann.FeedForwardTopology;
import com.alibaba.alink.operator.common.classification.ann.Topology;
import com.alibaba.alink.operator.common.classification.ann.TopologyModel;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.DeepFmDataFormat;
import com.alibaba.alink.operator.common.fm.FmLossUtils;
import com.alibaba.alink.operator.common.deepfm.BaseDeepFmTrainBatchOp.Task;
import com.alibaba.alink.operator.common.optim.subfunc.OptimVariable;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;
import com.alibaba.alink.operator.common.utils.FmOptimizerUtils;
import com.alibaba.alink.params.shared.linear.HasL1;
import com.alibaba.alink.params.shared.linear.HasL2;
import com.sun.org.apache.xpath.internal.operations.Bool;
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
                        + loss[3] / loss[1] + " mlp loss : " + loss[4] +
                        " time : " + (System.currentTimeMillis()
                        - oldTime));
                oldTime = System.currentTimeMillis();
            } else {
                System.out.println("step : " + context.getStepNo() + " loss : "
                        + loss[0] / loss[1] + "  mae : " + loss[2] / loss[1] + " mse : "
                        + loss[3] / loss[1] + " time : " + (System.currentTimeMillis() - oldTime));
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
            // for classification task: buffer[2] - AUC, buffer[3] - correct number
            double[] buffer = context.getObj(OptimVariable.lossAucAllReduce);
            if (buffer == null) {
                buffer = new double[5];
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
                double tmp = FmOptimizerUtils.fmCalcY(sample, factors.linearItems, factors.factors, factors.bias, dim).f0
                        + deepCalcY(factors, sample, dim).get(0);
                y[s] = 1.0 / (1 + Math.exp(-tmp));
            }

            double lossSum = 0.;
            for (int i = 0; i < y.length; i++) {
                double yTruth = labledVectors.get(i).f1;
                double l = lossFunc.l(yTruth, y[i]);
                lossSum += l;
            }

            if (this.task.equals(Task.REGRESSION)) {
                Tuple2<Double, Double> result = FmLossUtils.metricCalc.regression(labledVectors, y);
                buffer[2] = result.f0;
                buffer[3] = result.f1;
            } else {
                Tuple2<Double, Double> result = FmLossUtils.metricCalc.classification(labledVectors, y);
                buffer[2] = result.f0;
                buffer[3] = result.f1;
            }

            buffer[0] = lossSum;
            buffer[1] = y.length;
            buffer[4] = factors.dir.f1[1];      // MLP loss
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
            int size = factors.dir.f0.size();

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
                factors.dir.f0.set(i - start, buffer[i]);
            }
            factors.dir.f1[0] = buffer[start + size];
            factors.dir.f1[1] = buffer[start + size + 1];
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
        private double l1;
        private double l2;

        public UpdateLocalModel(int[] dim, double[] lambda, Params params) {
            this.lambda = lambda;
            this.dim = dim;
            this.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            this.learnRate = params.get(DeepFmTrainParams.LEARN_RATE);
            this.batchSize = params.get(DeepFmTrainParams.MINIBATCH_SIZE);
            this.l1 = params.get(HasL1.L_1);
            this.l2 = params.get(HasL2.L_2);

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
                int[] layerSize = new int[innerModel.layerSize.length - 2];
                for (int i = 0; i < layerSize.length; i++) {
                    layerSize[i] = innerModel.layerSize[i + 1];
                }
                sigmaGii = new DeepFmDataFormat(vectorSize, dim, layerSize,
                        innerModel.initialWeights, innerModel.dropoutRate, 0.0);
                context.putObj(OptimVariable.sigmaGii, sigmaGii);
            }

            updateFactors(labledVectors, innerModel, sigmaGii, weights, learnRate, context.getStepNo());

            // dim[0] - WITH_INTERCEPT, dim[1] - WITH_LINEAR_ITEM, dim[2] - NUM_FACTOR
            // prepare buffer vec for allReduce. the last element of vec is the weight Sum.
            double[] buffer = context.getObj(OptimVariable.factorAllReduce);
            int size = innerModel.dir.f0.size();
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
                buffer[i] = innerModel.dir.f0.get(i - start);
            }
            buffer[start + size] = innerModel.dir.f1[0];
            buffer[start + size + 1] = innerModel.dir.f1[1];
        }

        private void updateFactors(List<Tuple3<Double, Double, Vector>> labledVectors,
                                   DeepFmDataFormat factors,
                                   DeepFmDataFormat sigmaGii,
                                   double[] weights,
                                   double learnRate,
                                   int stepNum) {
            if (topology == null) {
                topology = FeedForwardTopology.multiLayerPerceptron(factors.layerSize,
                        true,
                        factors.dropoutRate);
            }

            double weightSum = 0.0;
            double loss = 0.0;

            for (int bi = 0; bi < batchSize; ++bi) {
                Tuple3<Double, Double, Vector> sample = labledVectors.get(rand.nextInt(labledVectors.size()));
                Vector vec = sample.f2;
                Tuple2<Double, double[]> yVx = FmOptimizerUtils.fmCalcY(vec, factors.linearItems,
                        factors.factors, factors.bias, dim);
                DenseVector yDeep = deepCalcY(factors, sample.f2, dim);
                Double yHat = 1.0 / (1 + Math.exp(-(yVx.f0 + yDeep.get(0))));

                double yTruth = sample.f1;
                double dldy = lossFunc.dldy(yTruth, yHat);
                dldy = dldy * yHat * (1 - yHat);

                // convert embedding data to DenseMatrix
                int[] indices = ((SparseVector) sample.f2).getIndices();
                DenseMatrix embeddings = new DenseMatrix(1, factors.factors.length * dim[2]);
                for (int i = 0; i < factors.factors.length; i++) {
                    for (int j = 0; j < dim[2]; j++) {
                        double embedding;
                        int finalI = i;
                        if (IntStream.of(indices).anyMatch(x -> x == finalI)) {
                            embedding = factors.factors[i][j] * sample.f2.get(finalI);
                        } else {
                            embedding = 0;
                        }
                        embeddings.set(0, i * dim[2] + j, embedding);
                    }
                }
                Boolean onehot = true;
                int[] layerSize = factors.layerSize;
                DenseMatrix labels = new DenseMatrix(1, onehot ? layerSize[layerSize.length - 1] : 1);
                if (onehot) {
                    Arrays.fill(labels.getData(), 0.);
                    labels.set(0, sample.f1.intValue(), 1.);
                } else {
                    throw new RuntimeException("Unsupported now.");
                    // labels.set(0, 0, sample.f1);
                }

                // update fm part params with Adagrad
                Tuple3<Tuple3<Double, double[], double[][]>, Tuple3<Double, double[], double[][]>, double[]> result
                        = calcGradient(Tuple3.of(sigmaGii.bias, sigmaGii.linearItems, sigmaGii.factors),
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
                if (sample.f2 instanceof SparseVector) {
                    ((SparseVector)(sample.f2)).setSize(factors.dir.f0.size());
                }
                weightSum += sample.f0;
                // update gradient
                if (topologyModel == null) {
                    topologyModel = topology.getModel(factors.dir.f0);
                } else {
                    topologyModel.resetModel(factors.dir.f0);
                }
                DenseVector oldCoefVector = factors.dir.f0;

                // calculate local gradient with Adagrad
                DenseVector grad = DenseVector.zeros(sigmaGii.dir.f0.size());
                for (int i = 0; i < grad.size(); i++) {
                    grad.set(i, 0.0);
                }

                topologyModel.computeGradient(embeddings, labels, grad);
                grad.scaleEqual(dldy);
                for (int i = 0; i < sigmaGii.dir.f0.size(); i++) {
                    sigmaGii.dir.f0.add(i, grad.get(i) * grad.get(i));
                    double eta = learnRate / Math.sqrt(sigmaGii.dir.f0.get(i) + eps);
                    factors.dir.f0.add(i, -eta * grad.get(i));
                }
                topologyModel.resetModel(factors.dir.f0);

                double tmpLoss = topologyModel.computeGradient(embeddings, labels, null);
                loss += tmpLoss * sample.f0;

                if (this.l1 != 0.) {
                    double[] coefArray = oldCoefVector.getData();
                    for (int i = 0; i < coefArray.length; i++) {
                        factors.dir.f0.add(i, Math.signum(coefArray[i] * this.l1));
                    }
                    loss += this.l1 * oldCoefVector.normL1();
                }
                if (this.l2 != 0.) {
                    factors.dir.f0.plusScaleEqual(oldCoefVector, this.l2);
                    loss += this.l2 * MatVecOp.dot(oldCoefVector, oldCoefVector);
                }
            }

            if (weightSum != 0.0) {
                //factors.dir.f0.scaleEqual(1.0 / weightSum);
                loss /= weightSum;
            }
            factors.dir.f1[0] = weightSum;
            factors.dir.f1[1] = loss;
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



    public static DenseVector deepCalcY(DeepFmDataFormat deepFmModel, Vector sampleFeature, int[] dim) {
        // convert and concatenate embeddings into 1 dimension
        int vecSize = deepFmModel.factors.length;
        int[] indices = ((SparseVector) sampleFeature).getIndices();
        DenseVector input = new DenseVector(vecSize * dim[2]);
        for (int i = 0; i < vecSize; i++) {
            for (int j = 0; j < dim[2]; j++) {
                int finalI = i;
                if (IntStream.of(indices).anyMatch(x -> x == finalI)) {
                    input.set(i * dim[2] + j, deepFmModel.factors[i][j] * sampleFeature.get(finalI));
                } else {
                    input.set(i * dim[2] + j, 0);
                }
            }
        }

        Topology topology = FeedForwardTopology.multiLayerPerceptron(deepFmModel.layerSize,
                true,
                deepFmModel.dropoutRate);
        TopologyModel topologyModel = topology.getModel(deepFmModel.dir.f0);
        DenseVector output = topologyModel.predict(input);

        return output;
    }

}
