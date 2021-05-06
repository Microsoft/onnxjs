// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {SessionHandler} from '../backend';
import {Session} from '../session';
import {MixedSessionHandler} from './mixed-session-handler';
import {WebGLBackend} from './backend-webgl';


export class MixedBackend extends WebGLBackend {
  createSessionHandler(context: Session.Context): SessionHandler {
    return new MixedSessionHandler(this, context);
  }
}
