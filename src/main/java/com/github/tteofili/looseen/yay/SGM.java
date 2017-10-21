/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package com.github.tteofili.looseen.yay;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.regex.Pattern;

import com.google.common.base.Splitter;
import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;
import org.apache.commons.math3.linear.RealMatrixPreservingVisitor;
import org.apache.commons.math3.linear.RealVector;

/**
 * A skip-gram neural network.
 * It learns its weights through backpropagation algorithm via (configurable) mini batch gradient descent applied to a collection of
 * hot encoded training samples.
 */
public class SGM {

    private final Configuration configuration;
    private final RectifierFunction rectifierFunction = new RectifierFunction();
    private final SoftmaxActivationFunction softmaxActivationFunction = new SoftmaxActivationFunction();

    /**
     * Each RealMatrix maps weights between two layers.
     * E.g.: weights[0] controls function mapping from layer 0 to layer 1.
     * If network has 4 units in layer 1 and 5 units in layer 2, then weights[0] will be of dimension 5x4, plus bias terms.
     * A network having layers with 3, 4 and 2 units each will have the following weights matrix dimensions:
     * - weights[0] : 4x3
     * - weights[1] : 2x4
     * <p>
     * the first row of weighs[0] matrix holds the weights of each neuron in the first neuron of the second layer,
     * the second row of weighs[0] holds the weights of each neuron in the second neuron of the second layer, etc.
     */
    private final RealMatrix[] weights;
    private final RealMatrix[] biases;
    private Sample[] samples;


    private SGM(Configuration configuration) {
        this.configuration = configuration;
        this.weights = initWeights();
        this.biases = initBiases();
    }

    private RealMatrix[] initBiases() {

        RealMatrix[] initialBiases = new RealMatrix[weights.length];

        for (int i = 0; i < initialBiases.length; i++) {
            double[] data = new double[weights[i].getRowDimension()];
            Arrays.fill(data, 0.01d);
            RealMatrix matrix = MatrixUtils.createRowRealMatrix(data);

            initialBiases[i] = matrix;
        }
        return initialBiases;
    }

    public RealMatrix[] getWeights() {
        return weights;
    }

    public List<String> getVocabulary() {
        return configuration.vocabulary;
    }

    private RealMatrix[] initWeights() {
        int[] conf = new int[]{configuration.inputs, configuration.vectorSize, configuration.outputs};
        int[] layers = new int[conf.length];
        System.arraycopy(conf, 0, layers, 0, layers.length);
        int weightsCount = layers.length - 1;

        RealMatrix[] initialWeights = new RealMatrix[weightsCount];

        for (int i = 0; i < weightsCount; i++) {
            RealMatrix matrix = MatrixUtils.createRealMatrix(layers[i + 1], layers[i]);

            UniformRealDistribution uniformRealDistribution = new UniformRealDistribution();
            double[] vs = uniformRealDistribution.sample(matrix.getRowDimension() * matrix.getColumnDimension());
            int r = 0;
            int c = 0;
            for (double v : vs) {
                matrix.setEntry(r % matrix.getRowDimension(), c % matrix.getColumnDimension(), v);
                r++;
                c++;
            }

            initialWeights[i] = matrix;
        }
        return initialWeights;

    }

    static double evaluate(SGM network) throws Exception {
        double cc = 0;
        double wc = 0;
        int window = network.configuration.window;
        List<String> vocabulary = network.getVocabulary();
        Collection<Integer> exps = new LinkedList<>();
        Collection<Integer> acts = new LinkedList<>();
        for (Sample sample : network.samples) {
            double[] inputs = sample.getInputs();
            int j = 0;
            for (int i = 0; i < window - 1; i++) {
                int le = inputs.length;
                int actualMax = getMaxIndex(network.predictOutput(inputs), j, j + le - 1);
                int expectedMax = getMaxIndex(sample.getOutputs(), j, j + le - 1);
                exps.add(expectedMax % le);
                acts.add(actualMax % le);
                j += le;
            }
            boolean c = true;
            for (Integer e : exps) {
                c &= acts.remove(e);
            }
            if (c) {
                cc++;
                String x = vocabulary.get(getMaxIndex(inputs, 0, inputs.length));
                StringBuilder y = new StringBuilder();
                for (int e : exps) {
                    if (y.length() > 0) {
                        y.append(" ");
                    }
                    y.append(vocabulary.get(e));
                }
                System.out.println("matched : " + x + " -> " + y);
            } else {
                wc++;
            }
            acts.clear();
            exps.clear();
            if (cc + wc > 2000) break;
        }
        return (cc / (wc + cc));
    }

    private static int getMaxIndex(double[] array, int start, int end) {
        double largest = array[start];
        int index = 0;
        for (int i = start + 1; i < end; i++) {
            if (array[i] >= largest) {
                largest = array[i];
                index = i;
            }
        }
        return index;
    }

    /**
     * predict network output given an input
     *
     * @param input the input
     * @return the output
     * @throws Exception
     */
    private double[] predictOutput(double[] input) throws Exception {

        RealMatrix hidden = rectifierFunction.applyMatrix(MatrixUtils.createRowRealMatrix(input).multiply(weights[0].transpose()).
                add(biases[0]));
        RealMatrix scores = hidden.multiply(weights[1].transpose()).add(biases[1]);

        RealMatrix probs = scores.copy();
        int len = scores.getColumnDimension() - 1;
        for (int d = 0; d < configuration.window - 1; d++) {
            int startColumn = d * len / (configuration.window - 1);
            RealMatrix subMatrix = scores.getSubMatrix(0, scores.getRowDimension() - 1, startColumn, startColumn + input.length);
            for (int sm = 0; sm < subMatrix.getRowDimension(); sm++) {
                probs.setSubMatrix(softmaxActivationFunction.applyMatrix(subMatrix.getRowMatrix(sm)).getData(), sm, startColumn);
            }
        }

        RealVector d = probs.getRowVector(0);
        return d.toArray();
    }


    // --- mini batch gradient descent ---

    /**
     * perform weights learning from the training examples using (configurable) mini batch gradient descent algorithm
     *
     * @param samples the training examples
     * @return the final cost with the updated weights
     * @throws Exception if BGD fails to converge or any numerical error happens
     */
    private double learnWeights(Sample... samples) throws Exception {

        int iterations = 0;

        double cost = Double.MAX_VALUE;

        int j = 0;

        // momentum
        RealMatrix vb = MatrixUtils.createRealMatrix(biases[0].getRowDimension(), biases[0].getColumnDimension());
        RealMatrix vb2 = MatrixUtils.createRealMatrix(biases[1].getRowDimension(), biases[1].getColumnDimension());
        RealMatrix vw = MatrixUtils.createRealMatrix(weights[0].getRowDimension(), weights[0].getColumnDimension());
        RealMatrix vw2 = MatrixUtils.createRealMatrix(weights[1].getRowDimension(), weights[1].getColumnDimension());

        long start = System.currentTimeMillis();
        int c = 1;
        RealMatrix x = MatrixUtils.createRealMatrix(configuration.batchSize, samples[0].getInputs().length);
        RealMatrix y = MatrixUtils.createRealMatrix(configuration.batchSize, samples[0].getOutputs().length);
        while (true) {

            int i = 0;
            for (int k = j * configuration.batchSize; k < j * configuration.batchSize + configuration.batchSize; k++) {
                Sample sample = samples[k % samples.length];
                x.setRow(i, sample.getInputs());
                y.setRow(i, sample.getOutputs());
                i++;
            }
            j++;

            long time = (System.currentTimeMillis() - start) / 1000;
            if (iterations % (1 + (configuration.maxIterations / 100)) == 0 && time > 60 * c) {
                c += 1;
//                System.out.println("cost: " + cost + ", accuracy: " + evaluate(this) + " after " + iterations + " iterations in " + (time / 60) + " minutes (" + ((double) iterations / time) + " ips)");
            }

            RealMatrix w0t = weights[0].transpose();
            RealMatrix w1t = weights[1].transpose();

            RealMatrix hidden = rectifierFunction.applyMatrix(x.multiply(w0t));
            hidden.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return value + biases[0].getEntry(0, column);
                }

                @Override
                public double end() {
                    return 0;
                }
            });
            RealMatrix scores = hidden.multiply(w1t);
            scores.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return value + biases[1].getEntry(0, column);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            RealMatrix probs = scores.copy();
            int len = scores.getColumnDimension() - 1;
            for (int d = 0; d < configuration.window - 1; d++) {
                int startColumn = d * len / (configuration.window - 1);
                RealMatrix subMatrix = scores.getSubMatrix(0, scores.getRowDimension() - 1, startColumn, startColumn + x.getColumnDimension());
                for (int sm = 0; sm < subMatrix.getRowDimension(); sm++) {
                    probs.setSubMatrix(softmaxActivationFunction.applyMatrix(subMatrix.getRowMatrix(sm)).getData(), sm, startColumn);
                }
            }

            RealMatrix correctLogProbs = MatrixUtils.createRealMatrix(x.getRowDimension(), 1);
            correctLogProbs.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return -Math.log(probs.getEntry(row, getMaxIndex(y.getRow(row))));
                }

                @Override
                public double end() {
                    return 0;
                }
            });
            double dataLoss = correctLogProbs.walkInOptimizedOrder(new RealMatrixPreservingVisitor() {
                private double d = 0;

                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public void visit(int row, int column, double value) {
                    d += value;
                }

                @Override
                public double end() {
                    return d;
                }
            }) / samples.length;

            double reg = 0d;
            reg += weights[0].walkInOptimizedOrder(new RealMatrixPreservingVisitor() {
                private double d = 0d;

                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public void visit(int row, int column, double value) {
                    d += Math.pow(value, 2);
                }

                @Override
                public double end() {
                    return d;
                }
            });
            reg += weights[1].walkInOptimizedOrder(new RealMatrixPreservingVisitor() {
                private double d = 0d;

                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public void visit(int row, int column, double value) {
                    d += Math.pow(value, 2);
                }

                @Override
                public double end() {
                    return d;
                }
            });

            double regLoss = 0.5 * configuration.regularizationLambda * reg;
            double newCost = dataLoss + regLoss;
            if (iterations == 0) {
//                System.out.println("started with cost = " + dataLoss + " + " + regLoss + " = " + newCost);
            }

            if (Double.POSITIVE_INFINITY == newCost) {
                throw new Exception("failed to converge at iteration " + iterations + " with alpha " + configuration.alpha + " : cost going from " + cost + " to " + newCost);
            } else if (iterations > 1 && (newCost < configuration.threshold || iterations > configuration.maxIterations)) {
                cost = newCost;
//                System.out.println("successfully converged after " + (iterations - 1) + " iterations (alpha:" + configuration.alpha + ",threshold:" + configuration.threshold + ") with cost " + newCost);
                break;
            } else if (Double.isNaN(newCost)) {
                throw new Exception("failed to converge at iteration " + iterations + " with alpha " + configuration.alpha + " : cost calculation underflow");
            }

            // update registered cost
            cost = newCost;

            // calculate the derivatives to update the parameters

            RealMatrix dscores = probs.copy();
            dscores.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return (y.getEntry(row, column) == 1 ? (value - 1) : value) / samples.length;
                }

                @Override
                public double end() {
                    return 0;
                }
            });


            // get derivative on second layer
            RealMatrix dW2 = hidden.transpose().multiply(dscores);

            // regularize dw2
            dW2.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return value + configuration.regularizationLambda * w1t.getEntry(row, column);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            RealMatrix db2 = MatrixUtils.createRealMatrix(biases[1].getRowDimension(), biases[1].getColumnDimension());
            dscores.walkInOptimizedOrder(new RealMatrixPreservingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public void visit(int row, int column, double value) {
                    db2.setEntry(0, column, db2.getEntry(0, column) + value);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            RealMatrix dhidden = dscores.multiply(weights[1]);
            dhidden.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return value < 0 ? 0 : value;
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            RealMatrix db = MatrixUtils.createRealMatrix(biases[0].getRowDimension(), biases[0].getColumnDimension());
            dhidden.walkInOptimizedOrder(new RealMatrixPreservingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public void visit(int row, int column, double value) {
                    db.setEntry(0, column, db.getEntry(0, column) + value);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            // get derivative on first layer
            RealMatrix dW = x.transpose().multiply(dhidden);

            // regularize
            dW.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                @Override
                public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                }

                @Override
                public double visit(int row, int column, double value) {
                    return value + configuration.regularizationLambda * w0t.getEntry(row, column);
                }

                @Override
                public double end() {
                    return 0;
                }
            });

            RealMatrix dWt = dW.transpose();
            RealMatrix dWt2 = dW2.transpose();


            if (configuration.useNesterovMomentum) {

                // update nesterov momentum
                final RealMatrix vbPrev = vb.copy();
                final RealMatrix vb2Prev = vb2.copy();
                final RealMatrix vwPrev = vw.copy();
                final RealMatrix vw2Prev = vw2.copy();

                vb.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * db.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                vb2.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * db2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                vw.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * dWt.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                vw2.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * dWt2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                // update bias
                biases[0].walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.mu * vbPrev.getEntry(row, column) + (1 + configuration.mu) * vb.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                biases[1].walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.mu * vb2Prev.getEntry(row, column) + (1 + configuration.mu) * vb2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                // update the weights
                weights[0].walkInOptimizedOrder(new RealMatrixChangingVisitor() {

                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.mu * vwPrev.getEntry(row, column) + (1 + configuration.mu) * vw.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                weights[1].walkInOptimizedOrder(new RealMatrixChangingVisitor() {

                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.mu * vw2Prev.getEntry(row, column) + (1 + configuration.mu) * vw2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });
            } else if (configuration.useMomentum) {
                // update momentum
                vb.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * db.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                vb2.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * db2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                vw.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * dWt.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                vw2.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return configuration.mu * value - configuration.alpha * dWt2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                // update bias
                biases[0].walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value + vb.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                biases[1].walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value + vb2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                // update the weights
                weights[0].walkInOptimizedOrder(new RealMatrixChangingVisitor() {

                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value + vw.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                weights[1].walkInOptimizedOrder(new RealMatrixChangingVisitor() {

                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value + vw2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });
            } else {
                // standard parameter update

                // update bias
                biases[0].walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.alpha * db.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                biases[1].walkInOptimizedOrder(new RealMatrixChangingVisitor() {
                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.alpha * db2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });


                // update the weights
                weights[0].walkInOptimizedOrder(new RealMatrixChangingVisitor() {

                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.alpha * dWt.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });

                weights[1].walkInOptimizedOrder(new RealMatrixChangingVisitor() {

                    @Override
                    public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

                    }

                    @Override
                    public double visit(int row, int column, double value) {
                        return value - configuration.alpha * dWt2.getEntry(row, column);
                    }

                    @Override
                    public double end() {
                        return 0;
                    }
                });
            }

            iterations++;
        }

        return cost;
    }

    private int getMaxIndex(double[] array) {
        double largest = array[0];
        int index = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] >= largest) {
                largest = array[i];
                index = i;
            }
        }
        return index;
    }

    public static SGM.Builder newModel() {
        return new Builder();
    }

    public Sample[] getSamples() {
        return samples;
    }

// --- skip gram neural network configuration ---

    private static class Configuration {
        // internal parameters
        int outputs;
        int inputs;

        List<String> vocabulary;

        // user controlled parameters
        int maxIterations;
        double alpha = 0.5d;
        double mu = 0.9d;
        double regularizationLambda = 0.03;
        double threshold = 0.0000000000004d;
        int vectorSize;
        int window;
        boolean useMomentum;
        boolean useNesterovMomentum;
        int batchSize;
        String text;
    }

    public static class Builder {
        private final Configuration configuration;

        public Builder() {
            this.configuration = new Configuration();
        }

        public Builder withBatchSize(int batchSize) {
            this.configuration.batchSize = batchSize;
            return this;
        }

        public Builder withWindow(int w) {
            this.configuration.window = w;
            return this;
        }

        public Builder fromText(String text) {
            this.configuration.text = text;
            return this;
        }

        public Builder withDimension(int d) {
            this.configuration.vectorSize = d;
            return this;
        }

        public Builder withAlpha(double alpha) {
            this.configuration.alpha = alpha;
            return this;
        }

        public Builder withLambda(double lambda) {
            this.configuration.regularizationLambda = lambda;
            return this;
        }

        public Builder withMu(double mu) {
            this.configuration.mu = mu;
            return this;
        }

        public Builder useMomentum(boolean useMomentum) {
            this.configuration.useMomentum = useMomentum;
            return this;
        }

        public Builder useNesterovMomentum() {
            this.configuration.useNesterovMomentum = true;
            return this;
        }

        public Builder withThreshold(double threshold) {
            this.configuration.threshold = threshold;
            return this;
        }

        public Builder withMaxIterations(int iterations) {
            this.configuration.maxIterations = iterations;
            return this;
        }

        public SGM build() throws Exception {
            Queue<List<byte[]>> fragments = getFragments(this.configuration.text, this.configuration.window);
            assert !fragments.isEmpty() : "could not read fragments for '" + this.configuration.text + "'";
            List<String> vocabulary = getVocabulary(fragments);
            assert !vocabulary.isEmpty() : "could not read vocabulary";
            this.configuration.vocabulary = vocabulary;

            Collection<HotEncodedSample> trainingSet = createTrainingSet(vocabulary, fragments, this.configuration.window);
            fragments.clear();
            if (this.configuration.maxIterations == 0) {
                this.configuration.maxIterations = trainingSet.size() * 100000;
            }

            if (this.configuration.batchSize == 0) {
                this.configuration.batchSize = trainingSet.size();
            }

            HotEncodedSample next = trainingSet.iterator().next();

            this.configuration.inputs = next.getInputs().length;
            this.configuration.outputs = next.getOutputs().length;

            SGM network = new SGM(configuration);
            network.samples = trainingSet.toArray(new Sample[trainingSet.size()]);
            network.learnWeights(network.samples);
            return network;
        }

        private List<String> getVocabulary(Queue<List<byte[]>> fragments) {
            List<String> vocabulary = new LinkedList<>();
            for (List<byte[]> fragment : fragments) {
                for (byte[] word : fragment) {
                    String s = new String(word);
                    if (!vocabulary.contains(s)) {
                        vocabulary.add(s);
                    }
                }
            }
            return vocabulary;
        }

        private Collection<HotEncodedSample> createTrainingSet(final List<String> vocabulary, Queue<List<byte[]>> fragments, int window) throws Exception {
            Collection<HotEncodedSample> samples = new LinkedList<>();
            List<byte[]> fragment;
            while ((fragment = fragments.poll()) != null) {
                List<byte[]> outputWords = new ArrayList<>(fragment.size() - 1);
                int inputIdx = fragment.size() / 2;
                byte[] inputWord = fragment.get(inputIdx);
                for (int k = 0; k < fragment.size(); k++) {
                    if (k != inputIdx) {
                        outputWords.add(fragment.get(k));
                    }
                }

                double[] doubles = new double[window - 1];
                for (int i = 0; i < doubles.length; i++) {
                    String o = new String(outputWords.get(i));
                    doubles[i] = (double) vocabulary.indexOf(o);
                }

                double[] inputs = new double[1];
                String x = new String(inputWord);
                inputs[0] = (double) vocabulary.indexOf(x);

                HotEncodedSample hotEncodedSample = new HotEncodedSample(inputs, doubles, vocabulary.size());
                samples.add(hotEncodedSample);
            }

            return samples;
        }

        private Queue<List<byte[]>> getFragments(String text, int w) throws IOException {
            Queue<List<byte[]>> fragments = new ConcurrentLinkedDeque<>();

            Splitter splitter = Splitter.on(Pattern.compile("[\\n\\s]")).omitEmptyStrings().trimResults();

            addFragments(text, w, fragments, splitter);
            return fragments;

        }

        private void addFragments(String text, int w, Queue<List<byte[]>> fragments, Splitter splitter) {
            ByteBuffer buffer = ByteBuffer.wrap(text.getBytes());
            try {
                StringBuffer line = new StringBuffer();
                for (int i = 0; i < buffer.limit(); i++) {
                    char ch = ((char) buffer.get());
                    if (ch == '\r' || ch == '\n' || i + 1 == buffer.limit()) {
                        // create fragments for this line
                        String string = cleanString(line.toString());
                        List<String> split = splitter.splitToList(string);
                        int splitSize = split.size();
                        if (splitSize >= w) {
                            for (int j = 0; j < splitSize - w; j++) {
                                List<byte[]> fragment = new ArrayList<>(w);
                                String str = split.get(j);
                                fragment.add(str.getBytes());
                                for (int k = 1; k < w; k++) {
                                    String s = split.get(k + j);
                                    fragment.add(s.getBytes());
                                }
                                // TODO : this has to be used to re-use the tokens that have not been consumed in next iteration
                                fragments.add(fragment);
                            }
                        }
                        line = new StringBuffer();
                    } else {
                        line.append(ch);
                    }
                }

            } finally {
                buffer.clear();
            }
        }

        private String cleanString(String s) {
            return s.toLowerCase().replaceAll("\\.", " \\.").replaceAll("\\;", " \\;").replaceAll("\\,", " \\,").replaceAll("\\:", " \\:").replaceAll("\\-\\s", "").replaceAll("\\\"", " \\\"");
        }
    }
}