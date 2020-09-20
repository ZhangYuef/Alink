package com.alibaba.alink.pipeline.classification;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.junit.Test;

public class UniversalTest {
    @Test
    public void Test() throws Exception {
        BinomialDistribution bionimialDistribution = new BinomialDistribution(1, 1 - 0.);

        for (int i = 0; i < 10; i++) {

            System.out.println(bionimialDistribution.sample());

        }
    }
}
