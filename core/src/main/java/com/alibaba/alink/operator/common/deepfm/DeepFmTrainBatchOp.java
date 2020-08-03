package com.alibaba.alink.operator.common.deepfm;

import com.alibaba.alink.common.MLEnvironment;
import com.alibaba.alink.common.linalg.DenseVector;
import com.alibaba.alink.common.linalg.Vector;
import com.alibaba.alink.common.model.ModelParamName;
import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.common.classification.ann.Topology;
import com.alibaba.alink.operator.common.optim.DeepFmOptimizer;
import com.alibaba.alink.params.recommendation.DeepFmTrainParams;

import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.ml.api.misc.param.Params;

import java.util.Random;

/**
 * DeepFM model training.
 */
public class DeepFmTrainBatchOp<T extends DeepFmTrainBatchOp<T>> extends BaseDeepFmTrainBatchOp<T> {
    private static final long serialVersionUID = -7121049631706290487L;

    private DenseVector initialWeights;
    /**
     * construct function.
     *
     * @param params parameters needed by training process.
     * @param task   Fm task: maybe "classification" or "regression".
     */
    public DeepFmTrainBatchOp(Params params, String task) {
        super(params.set(ModelParamName.TASK, task));
    }

    /**
     * construct function.
     *
     * @param task
     */
    public DeepFmTrainBatchOp(String task) {
        super(new Params().set(ModelParamName.TASK, task));
    }

    /**
     * optimize function.
     *
     * @param trainData training Data.
     * @param vecSize   vector size.
     * @param params    parameters.
     * @param dim       dimension.
     * @param topology  topology for multi-layer perception network
     * @param session   environment.
     * @return
     */
    @Override
    protected DataSet<DeepFmDataFormat> optimize(DataSet<Tuple3<Double, Double, Vector>> trainData,
                                                 DataSet<Integer> vecSize,
                                                 final Params params,
                                                 final int[] dim,
                                                 final Topology topology,
                                                 MLEnvironment session) {
        final double initStdev = params.get(DeepFmTrainParams.INIT_STDEV);
        final DenseVector initialWeights = params.get(DeepFmTrainParams.INITIAL_WEIGHTS);

        DataSet<DeepFmDataFormat> initFactors = initFactorsModel(vecSize, dim, initStdev);
        DataSet<DenseVector> initMlpWeights = initMlpModel(trainData, topology, initialWeights);

        DeepFmOptimizer optimizer = new DeepFmOptimizer(trainData, topology, params);
        optimizer.setWithInitFactors(initFactors);
        optimizer.setWithInitMlpWeights(initMlpWeights);

        return optimizer.optimize();
    }

    private DataSet<DeepFmDataFormat> initFactorsModel(DataSet<Integer> vecSize, int[] dim,double initStdev) {
        // TODO: does this need getExecutionEnvironmentFromDataSets explicitly?
        return vecSize.map(new RichMapFunction<Integer, DeepFmDataFormat>() {
            private static final long serialVersionUID = 5898149131657343503L;

            @Override
            public DeepFmDataFormat map(Integer value) throws Exception {
                DeepFmDataFormat innerModel = new DeepFmDataFormat(value, dim, initStdev);

                return innerModel;
            }
        });
    }

    private DataSet<DenseVector> initMlpModel(DataSet<?> inputRel, Topology topology, DenseVector initialWeights) {
        if (initialWeights != null) {
            if (initialWeights.size() != topology.getWeightSize()) {
                throw new RuntimeException("Invalid initial weights, size mismatch");
            }
            return BatchOperator.getExecutionEnvironmentFromDataSets(inputRel).fromElements(this.initialWeights);
        } else {
            return BatchOperator.getExecutionEnvironmentFromDataSets(inputRel).fromElements(0)
                    .map(new RichMapFunction<Integer, DenseVector>() {
                        final double initStdev = 0.05;
                        final long seed = 1L;
                        transient Random random;

                        @Override
                        public void open(Configuration parameters) throws Exception {
                            random = new Random(seed);
                        }

                        @Override
                        public DenseVector map(Integer value) throws Exception {
                            DenseVector weights = DenseVector.zeros(topology.getWeightSize());
                            for (int i = 0; i < weights.size(); i++) {
                                weights.set(i, random.nextGaussian() * initStdev);
                            }
                            return weights;
                        }
                    })
                    .name("init_weights");
        }
    }

}
