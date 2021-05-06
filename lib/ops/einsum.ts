// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Einsum implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  prepareRun(inputs: Tensor[]) {
    const dimensionSizeMap: {[name: string]: number} = {};
    this.matchInputs(inputs, dimensionSizeMap);
    const outputShape = this.calculateOutputSize(dimensionSizeMap);

    let i = 0;
    const sizes = [];
    const nameToId: {[name: string]: number} = {};
    const idToName: {[id: number]: string} = {};

    for (const name in dimensionSizeMap) {
      sizes.push(dimensionSizeMap[name]);
      nameToId[name] = i;
      idToName[i] = name;
      i++;
    }

    const outputIndices: number[] = [];
    const inputIndices: number[][] = [];
    for (const outputName of this.outputNames) {
      outputIndices.push(nameToId[outputName]);
    }
    for (let i = 0; i < this.inputs.length; i++) {
      const indices = [];
      for (const inputName of this.inputNames[i]) {
        indices.push(nameToId[inputName]);
      }
      inputIndices.push(indices);
    }

    return {outputShape, sizes, outputIndices, inputIndices};
  }

  initialize(attributes: Attribute): void {
    this.equation = attributes.getString('equation');
    const split = this.equation.split('->');
    this.lhs = split[0].trim();
    if (split.length === 2) {
      this.rhs = split[1].trim();
      this.implicit = false;
    } else {
      this.implicit = true;
    }

    const lhsSplit = this.lhs.split(',');
    this.inputs = lhsSplit.map(v => v.trim());

    for (let i = 0; i < this.inputs.length; i++) {
      this.inputNames.push([]);
      this.parseEquationPart(this.inputs[i], this.inputNames[i]);
    }

    if (this.rhs) {
      this.parseEquationPart(this.rhs, this.outputNames);
    }
  }

  private parseEquationPart(part: string, indices: string[]) {
    for (let i = 0; i < part.length; i++) {
      const char = part.charAt(i);

      if (char === '.') {
        throw new Error('Use of ellipsis (...) in einsum not yet supported');
      }

      indices.push(char);
    }
  }

  protected matchInputs(inputs: Tensor[], dimensionSizeMap: {[name: string]: number}) {
    for (let i = 0; i < inputs.length; i++) {
      this.matchDimensions(this.inputNames[i], inputs[i].dims, dimensionSizeMap);
    }
  }

  protected calculateOutputSize(dimensionSizeMap: {[name: string]: number}): number[] {
    const result: number[] = [];
    for (let i = 0; i < this.outputNames.length; i++) {
      result.push(dimensionSizeMap[this.outputNames[i]]);
    }
    return result;
  }

  checkInputs(inputs: Tensor[]): boolean {
    const dimensionMap: {[id: string]: number} = {};

    if (inputs.length !== this.inputs.length) {
      return false;
    }

    for (let i = 0; i < inputs.length; i++) {
      if (!this.matchDimensions(this.inputNames[i], inputs[i].dims, dimensionMap)) {
        return false;
      }
    }

    return this.checkInputTypes(inputs);
  }

  protected matchDimensions(indices: string[], inputDims: readonly number[], dimensionMap: {[id: string]: number}):
      boolean {
    for (let j = 0; j < indices.length; j++) {
      const ix = indices[j];
      if (dimensionMap[ix] && dimensionMap[ix] !== inputDims[j]) {
        return false;
      } else if (!dimensionMap[ix]) {
        dimensionMap[ix] = inputDims[j];
      }
    }

    return true;
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    const allowedTypes = ['float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'];

    if (inputs.find((v) => allowedTypes.indexOf(v.type) === -1) !== undefined) {
      return false;
    }

    const types = inputs.map(v => v.type);
    if (types.find(v => v !== types[0]) !== undefined) {
      return false;
    }

    return true;
  }

  protected equation: string;
  protected lhs: string;
  protected rhs?: string;

  protected inputs: string[] = [];

  // The i-th string[] Maps from input axis i to general axis id
  protected inputNames: string[][] = [];

  // Maps from output axis to general axis id
  protected outputNames: string[] = [];

  protected implicit: boolean;
}
