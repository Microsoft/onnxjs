// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class CumSum implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.exclusive = attributes.getInt('exclusive', 0) === 1;
    this.reverse = attributes.getInt('reverse', 0) === 1;
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || inputs.length !== 2) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[1].type !== 'int32' || inputs[1].dims.length !== 1) {
      return false;
    }

    return true;
  }

  protected exclusive: boolean;
  protected reverse: boolean;
}
