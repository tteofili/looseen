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

import org.apache.commons.math3.linear.RealMatrix;

/**
 * An activation function AF : S -* S receives a signal and generates a new signal.
 * An activation function AF has horizontal asymptotes at 0 and 1 and a non
 * decreasing first derivative AF' with AF and AF' both being computable.
 * These are usually used in neurons in order to propagate the "signal"
 * throughout the whole network.
 */
public interface ActivationFunction {

  /**
   * Apply this <code>ActivationFunction</code> to the given matrix of signals, generating a new matrix of transformed
   * signals.
   *
   * @param weights the matrix of weights the activation should be applied to
   * @return the output signal generated as a {@link RealMatrix}
   */
  RealMatrix applyMatrix(RealMatrix weights);

}
