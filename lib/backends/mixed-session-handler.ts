// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {WebGLSessionHandler} from './webgl/session-handler';
import {Graph} from '../graph';
import {OpSet, resolveOperator} from '../opset';
import {Operator} from '../operators';
import {CPU_OP_RESOLVE_RULES} from './cpu/op-resolve-rules';
import {Logger} from '../instrument';

export class MixedSessionHandler extends WebGLSessionHandler {
    resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>): Operator {
        try {
            return super.resolve(node, opsets);
        } catch (e) {
            Logger.warning(
                'MixedSessionHandler',
                `Unable to initialize operator '${node.opType}' with webgl. trying with cpu...`);
            const op = resolveOperator(node, opsets, CPU_OP_RESOLVE_RULES);
            op.initialize(node.attributes);
            return op;
        }
    }
}