import numpy as np
import popart
import logging_util

logger = logging_util.get_basic_logger('TEST_UTILS')

def helper_run(builder, input_tensors, input_values, output_tensors, batches_per_step, training=False, loss=None, optimizer=None, outline=False, seed=None):
    output_tensors_return_types = {}
    for output_tensor in output_tensors:
        output_tensors_return_types[output_tensor] = popart.AnchorReturnType("ALL")
    proto = builder.getModelProto()        
    data_flow = popart.DataFlow(batches_per_step, output_tensors_return_types)
    device_info=popart.DeviceManager().acquireAvailableDevice(1)
    user_options = popart.SessionOptions()
    if outline:
        user_options.enableOutlining = True
        user_options.outlineThreshold = -np.inf
    if not training:
        session = popart.InferenceSession(
            fnModel=proto,
            dataFlow=data_flow,
            deviceInfo=device_info,
            userOptions=user_options
        )
    else:
        session = popart.TrainingSession(
            fnModel=proto,
            dataFlow=data_flow,
            loss=loss,
            optimizer=optimizer,
            deviceInfo=device_info,
            userOptions=user_options
        )

    session.prepareDevice()
    if seed is not None:
        logger.debug("Random seed set up to {}".format(seed))
        session.setRandomSeed(seed)
    if training:
        session.weightsFromHost()
    anchors = session.initAnchorArrays()
    input_anchors = {}
    for i in range(len(input_tensors)):
        input_anchors[input_tensors[i]] = input_values[i]
    stepio = popart.PyStepIO(input_anchors, anchors)
    session.run(stepio)
    output_values = []
    for output_tensor in output_tensors:
        output_values.append(anchors[output_tensor])
    return output_values