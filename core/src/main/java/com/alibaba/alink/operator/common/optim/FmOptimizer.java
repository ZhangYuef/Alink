package com.alibaba.alink.operator.common.optim;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import com.alibaba.alink.common.comqueue.ComContext;
import com.alibaba.alink.common.comqueue.CompareCriterionFunction;
import com.alibaba.alink.common.comqueue.CompleteResultFunction;
import com.alibaba.alink.common.comqueue.ComputeFunction;
import com.alibaba.alink.common.comqueue.IterativeComQueue;
import com.alibaba.alink.common.comqueue.communication.AllReduce;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.common.utils.JsonConverter;
import com.alibaba.alink.operator.common.fm.BaseFmTrainBatchOp.FmDataFormat;
import com.alibaba.alink.operator.common.fm.FmLossUtils;
import com.alibaba.alink.operator.common.fm.BaseFmTrainBatchOp.Task;
import com.alibaba.alink.operator.common.optim.subfunc.OptimVariable;
import com.alibaba.alink.params.recommendation.FmTrainParams;
import com.alibaba.alink.operator.common.utils.FmOptimizerUtils;
import com.alibaba.alink.operator.common.fm.FmLossUtils;

import org.apache.flink.api.common.functions.RichMapPartitionFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.ml.api.misc.param.Params;
import org.apache.flink.types.Row;
import org.apache.flink.util.Collector;

import static com.alibaba.alink.operator.common.utils.FmOptimizerUtils.calcGradient;

/**
 * Fm optimizer.
 */
public class FmOptimizer {
    private Params params;
    private DataSet<Tuple3<Double, Double, Vector>> trainData;
    private int[] dim;
    protected DataSet<FmDataFormat> fmModel = null;
    private double[] lambda;

    /**
     * construct function.
     *
     * @param trainData train data.
     * @param params    parameters for optimizer.
     */
    public FmOptimizer(DataSet<Tuple3<Double, Double, Vector>> trainData, Params params) {
        this.params = params;
        this.trainData = trainData;

        this.dim = new int[3];
        dim[0] = params.get(FmTrainParams.WITH_INTERCEPT) ? 1 : 0;
        dim[1] = params.get(FmTrainParams.WITH_LINEAR_ITEM) ? 1 : 0;
        dim[2] = params.get(FmTrainParams.NUM_FACTOR);

        this.lambda = new double[3];
        lambda[0] = params.get(FmTrainParams.LAMBDA_0);
        lambda[1] = params.get(FmTrainParams.LAMBDA_1);
        lambda[2] = params.get(FmTrainParams.LAMBDA_2);
    }

    /**
     * initialize fmModel.
     */
    public void setWithInitFactors(DataSet<FmDataFormat> model) {
        this.fmModel = model;
    }

    /**
     * optimize Fm problem.
     *
     * @return fm model.
     */
    public DataSet<FmDataFormat> optimize() {
        DataSet<Row> model = new IterativeComQueue()
            .initWithPartitionedData(OptimVariable.fmTrainData, trainData)
            .initWithBroadcastData(OptimVariable.fmModel, fmModel)
            .add(new UpdateLocalModel(dim, lambda, params))
            .add(new AllReduce(OptimVariable.factorAllReduce))
            .add(new UpdateGlobalModel(dim))
            .add(new CalcLossAndEvaluation(dim, params.get(ModelParamName.TASK)))
            .add(new AllReduce(OptimVariable.lossAucAllReduce))
            .setCompareCriterionOfNode0(new FmIterTermination(params))
            .closeWith(new OutputFmModel())
            .setMaxIter(Integer.MAX_VALUE)
            .exec();
        return model.mapPartition(new ParseRowModel());
    }

    /**
     * Termination class of fm iteration.
     */
    public static class FmIterTermination extends CompareCriterionFunction {
        private static final long serialVersionUID = 2410437704683855923L;
        private double oldLoss;
        private double epsilon;
        private Task task;
        private int maxIter;
        private int batchSize;
        private Long oldTime;

        public FmIterTermination(Params params) {
            this.maxIter = params.get(FmTrainParams.NUM_EPOCHS);
            this.epsilon = params.get(FmTrainParams.EPSILON);
            this.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            this.batchSize = params.get(FmTrainParams.MINIBATCH_SIZE);
            this.oldTime = System.currentTimeMillis();
        }

        @Override
        public boolean calc(ComContext context) {
            int numSamplesOnNode0
                = ((List<Tuple3<Double, Double, Vector>>)context.getObj(OptimVariable.fmTrainData)).size();

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
        private static final long serialVersionUID = 1276524768860519162L;
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
            double[] buffer = context.getObj(OptimVariable.lossAucAllReduce);
            // prepare buffer vec for allReduce. the last element of vec is the weight Sum.
            if (buffer == null) {
                buffer = new double[4];
                context.putObj(OptimVariable.lossAucAllReduce, buffer);
            }
            List<Tuple3<Double, Double, Vector>> labledVectors = context.getObj(OptimVariable.fmTrainData);
            if (this.y == null) {
                this.y = new double[labledVectors.size()];
            }
            // get fmModel from static memory.
            FmDataFormat factors = ((List<FmDataFormat>)context.getObj(OptimVariable.fmModel)).get(0);
            Arrays.fill(y, 0.0);
            for (int s = 0; s < labledVectors.size(); s++) {
                Vector sample = labledVectors.get(s).f2;
                y[s] = FmOptimizerUtils.fmCalcY(sample, factors.linearItems, factors.factors,
                        factors.bias, dim).f0;
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
        }
    }

    /**
     * Update global fm model.
     */
    public static class UpdateGlobalModel extends ComputeFunction {
        private static final long serialVersionUID = 4584059654350995646L;
        private int[] dim;

        public UpdateGlobalModel(int[] dim) {
            this.dim = dim;
        }

        @Override
        public void calc(ComContext context) {
            double[] buffer = context.getObj(OptimVariable.factorAllReduce);
            FmDataFormat sigmaGii = context.getObj(OptimVariable.sigmaGii);

            FmDataFormat factors = ((List<FmDataFormat>)context.getObj(OptimVariable.fmModel)).get(0);

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
     * Update local fm model.
     */
    public static class UpdateLocalModel extends ComputeFunction {

        private static final long serialVersionUID = 5331512619834061299L;
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

        public UpdateLocalModel(int[] dim, double[] lambda, Params params) {
            this.lambda = lambda;
            this.dim = dim;
            this.task = Task.valueOf(params.get(ModelParamName.TASK).toUpperCase());
            this.learnRate = params.get(FmTrainParams.LEARN_RATE);
            this.batchSize = params.get(FmTrainParams.MINIBATCH_SIZE);
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
            ArrayList<Tuple3<Double, Double, Vector>> labledVectors = context.getObj(OptimVariable.fmTrainData);

            if (batchSize == -1) {
                batchSize = labledVectors.size();
            }

            // get fmModel from static memory.
            FmDataFormat sigmaGii = context.getObj(OptimVariable.sigmaGii);
            FmDataFormat innerModel = ((List<FmDataFormat>)context.getObj(OptimVariable.fmModel)).get(0);
            double[] weights = context.getObj(OptimVariable.weights);
            if (weights == null) {
                vectorSize = (innerModel.factors != null) ? innerModel.factors.length : innerModel.linearItems.length;
                weights = new double[vectorSize];
                context.putObj(OptimVariable.weights, weights);
            } else {
                Arrays.fill(weights, 0.);
            }

            if (sigmaGii == null) {
                sigmaGii = new FmDataFormat(vectorSize, dim, 0.0);
                context.putObj(OptimVariable.sigmaGii, sigmaGii);
            }

            updateFactors(labledVectors, innerModel, learnRate, sigmaGii, weights);

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

        /**
         * Update FM model parameters by Adagrad optimization.
         */
        private void updateFactors(List<Tuple3<Double, Double, Vector>> labledVectors,
                                   FmDataFormat factors,
                                   double learnRate,
                                   FmDataFormat sigmaGii,
                                   double[] weights) {
            for (int bi = 0; bi < batchSize; ++bi) {
                Tuple3<Double, Double, Vector> sample = labledVectors.get(rand.nextInt(labledVectors.size()));
                Vector vec = sample.f2;
                Tuple2<Double, double[]> yVx = FmOptimizerUtils.fmCalcY(vec, factors.linearItems, factors.factors,
                        factors.bias, dim);
                double yTruth = sample.f1;
                double dldy = lossFunc.dldy(yTruth, yVx.f0);

                // update fmModel
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

            }
        }
    }

    /**
     * Output fm model with row format.
     */
    public static class OutputFmModel extends CompleteResultFunction {

        private static final long serialVersionUID = 727259322769437038L;

        @Override
        public List<Row> calc(ComContext context) {
            if (context.getTaskId() != 0) {
                return null;
            }
            FmDataFormat factors = ((List<FmDataFormat>)context.getObj(OptimVariable.fmModel)).get(0);
            List<Row> model = new ArrayList<>();
            model.add(Row.of(JsonConverter.toJson(factors)));
            return model;
        }
    }

    public class ParseRowModel extends RichMapPartitionFunction<Row, FmDataFormat> {
        private static final long serialVersionUID = -2078134573230730223L;

        @Override
        public void mapPartition(Iterable<Row> iterable,
                                 Collector<FmDataFormat> collector) throws Exception {
            int taskId = getRuntimeContext().getIndexOfThisSubtask();
            if (taskId == 0) {
                for (Row row : iterable) {

                    FmDataFormat factor = JsonConverter.fromJson((String)row.getField(0), FmDataFormat.class);
                    collector.collect(factor);
                }
            }
        }
    }
}
