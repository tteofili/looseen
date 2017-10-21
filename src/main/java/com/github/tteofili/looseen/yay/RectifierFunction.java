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
import org.apache.commons.math3.linear.RealMatrixChangingVisitor;

/**
 * Rectifier (aka ReLU) activation function
 */
public class RectifierFunction implements ActivationFunction {
  @Override
  public RealMatrix applyMatrix(RealMatrix weights) {
    RealMatrix matrix = weights.copy();
    matrix.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
      @Override
      public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

      }

      @Override
      public double visit(int row, int column, double value) {
        return Math.max(0, value);
      }

      @Override
      public double end() {
        return 0;
      }
    });
    return matrix;
  }
}
