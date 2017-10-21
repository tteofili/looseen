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

import java.util.Arrays;

/**
 * an hot-encoded {@link Sample}
 */
public class HotEncodedSample extends Sample {

  private double[] expandedInputs = null;
  private double[] expandedOutputs = null;

  private final int vocabularySize;

  public HotEncodedSample(double[] inputs, double[] outputs, int vocabularySize) {
    super(inputs, outputs);
    this.vocabularySize = vocabularySize;
  }

  @Override
  public double[] getInputs() {
    if (expandedInputs == null) {
      double[] inputs = new double[this.inputs.length * vocabularySize];
      int i = 0;
      for (double d : this.inputs) {
        double[] currentInput = hotEncode((int) d);
        System.arraycopy(currentInput, 0, inputs, i, currentInput.length);
        i += vocabularySize;
      }
      expandedInputs = inputs;
    }
    return expandedInputs;
  }

  @Override
  public double[] getOutputs() {
    if (expandedOutputs == null) {
      double[] outputs = new double[this.outputs.length * vocabularySize];
      int i = 0;
      for (double d : this.outputs) {
        double[] currentOutput = hotEncode((int) d);
        System.arraycopy(currentOutput, 0, outputs, i, currentOutput.length);
        i += vocabularySize;
      }
      expandedOutputs = outputs;
    }
    return expandedOutputs;
  }

  private double[] hotEncode(int index) {
    double[] vector = new double[vocabularySize];
    Arrays.fill(vector, 0d);
    vector[index] = 1d;
    return vector;
  }
}
