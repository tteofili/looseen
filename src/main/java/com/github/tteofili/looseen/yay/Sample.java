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

/**
 * a training example
 */
public class Sample {

  protected final double[] inputs;
  protected final double[] outputs;

  public Sample(double[] inputs, double[] outputs) {
    this.inputs = inputs;

    this.outputs = outputs;
  }

  /**
   * get the inputs as a double vector
   *
   * @return a double array
   */
  public double[] getInputs() {
    double[] result = new double[inputs.length + 1];
    result[0] = 1d;
    System.arraycopy(inputs, 0, result, 1, inputs.length);
    return result;
  }

  /**
   * get the outputs as a double vector
   *
   * @return a double array
   */
  public double[] getOutputs() {
    return outputs;
  }

}
