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
import org.apache.commons.math3.linear.RealMatrixPreservingVisitor;

/**
 * Softmax activation function
 */
public class SoftmaxActivationFunction implements ActivationFunction {

  public RealMatrix applyMatrix(RealMatrix weights) {
    RealMatrix matrix = weights.copy();
    final double finalD = expDen(matrix);
    matrix.walkInOptimizedOrder(new RealMatrixChangingVisitor() {
      @Override
      public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

      }

      @Override
      public double visit(int row, int column, double value) {
        return Math.exp(value) / finalD;
      }

      @Override
      public double end() {
        return 0;
      }
    });
    return matrix;
  }

  private double expDen(RealMatrix matrix) {
    return matrix.walkInOptimizedOrder(new RealMatrixPreservingVisitor() {
      private double d1 = 0d;

      @Override
      public void start(int rows, int columns, int startRow, int endRow, int startColumn, int endColumn) {

      }

      @Override
      public void visit(int row, int column, double value) {
        d1 += Math.exp(value);
      }

      @Override
      public double end() {
        return d1;
      }
    });
  }

}
