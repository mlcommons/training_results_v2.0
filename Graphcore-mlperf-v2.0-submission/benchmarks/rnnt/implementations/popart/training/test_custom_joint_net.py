import argparse
import numpy as np
import ctypes
import popart
import custom_op_utils
import logging_util
from test_utils import helper_run
from transducer_train import _get_popart_type as _get_popart_type
import transducer_blocks

logger = logging_util.get_basic_logger('TEST_CUSTOM_OP_JOINT_NET')

class JointNetwork(transducer_blocks.Block):
    def __init__(
            self,
            builder,
            transcription_out_len,
            joint_n_hid,
            num_symbols,
            joint_dropout,
            dtype=np.float32,
            transcription_out_split_size=15,
            shift_labels_by_one=True,
            custom_op=False,
            custom_dropout=False,
            output_dropout_mask=False,
            no_recompute_fc_in=False,
            no_recompute_fc_out=False):

        super(JointNetwork, self).__init__(builder, dtype, block_name="joint_network")
        self.joint_n_hid = joint_n_hid
        self.num_symbols = num_symbols
        self.joint_dropout = joint_dropout

        self.transcription_out_split_size = transcription_out_split_size
        if transcription_out_split_size > 0:
            self.transcription_splitter = transducer_blocks.Split(builder,
                                                            total_size=transcription_out_len,
                                                            split_size=transcription_out_split_size,
                                                            split_axis=1,
                                                            dtype=dtype,
                                                            block_name="joint_net_transcription_splitter")
        else:
            self.transcription_splitter = None
            
        self.joint_out_fc = transducer_blocks.RHSLinear(builder,
                                                        joint_n_hid,
                                                        num_symbols,
                                                        dtype=dtype,
                                                        block_name='joint_net_out_fc')

        self.custom_op = custom_op
        self.shift_labels_by_one = shift_labels_by_one
        self.custom_dropout = custom_dropout
        self.output_dropout_mask = output_dropout_mask
        self.no_recompute_fc_in = no_recompute_fc_in
        self.no_recompute_fc_out = no_recompute_fc_out

    def __call__(self, *args):
        return self.__build_graph(*args)

    def get_compact_log_probs(self, transcription_out_split, prediction_out, targets, target_lens):

        builder = self.builder
        num_dropout_outputs = 2 if self.output_dropout_mask else 1

        if self.custom_op:
            co_inputs = [transcription_out_split, prediction_out, self.joint_out_fc.weights, self.joint_out_fc.biases, targets, target_lens]
            num_outputs = 1
            if self.no_recompute_fc_in:
                num_outputs += 1
            if self.no_recompute_fc_out:
                num_outputs += 1
            if self.output_dropout_mask:
                num_outputs += 1
            joint_net_output_co = builder.customOp(opName = "GetCompactLogProbs",
                                        opVersion=1,
                                        domain = "com.acme",
                                        inputs = co_inputs,
                                        attributes = {
                                            "dropout_rate": self.joint_dropout,
                                            "shift_labels_by_one": self.shift_labels_by_one,
                                        },
                                        numOutputs = num_outputs)
            compact_log_probs = joint_net_output_co[0]
            out_idx = 1
            if self.no_recompute_fc_in:
                out_idx += 1
            if self.no_recompute_fc_out:
                out_idx += 1
            if self.output_dropout_mask:
                mask = joint_net_output_co[out_idx]
                out_idx += 1
            assert(out_idx == num_outputs)
            
            joint_net_output = [compact_log_probs]
            if self.output_dropout_mask:
                joint_net_output.append(mask)

        elif self.custom_dropout:
            joint_out_split = builder.aiOnnx.add([transcription_out_split, prediction_out])
            relu = builder.aiOnnx.relu([joint_out_split])
            
            cd_inputs = [relu]
                
            out_d = builder.customOp(
                opName = "CustomDropout",
                opVersion = 1,
                domain = "com.acme",
                inputs = cd_inputs,
                attributes = {
                    "dropout_rate": self.joint_dropout
                },
                numOutputs = num_dropout_outputs)
            dropout = out_d[0]
            if self.output_dropout_mask:
                mask = out_d[1]           
            joint_out_split = self.joint_out_fc(dropout, force_recompute=True)
            
            if self.shift_labels_by_one:
                one = self.builder.aiOnnx.constant(np.array([1]).astype(np.int32))
                targets = self.builder.aiOnnx.add([targets, one])

            compact_log_probs = builder.customOp(
                opName="SparseLogSoftmax",
                opVersion=1,
                domain="com.acme",
                inputs=[joint_out_split, targets, target_lens],
                attributes={},
                numOutputs=1,
            )[0]
            joint_net_output = [compact_log_probs]
            if self.output_dropout_mask:
                joint_net_output.append(mask)
        else:
            joint_out_split = builder.aiOnnx.add([transcription_out_split, prediction_out])
            relu = builder.aiOnnx.relu([joint_out_split])
            out_d = self.builder.aiOnnx.dropout([relu], num_dropout_outputs, self.joint_dropout)
            dropout = out_d[0]
            if self.output_dropout_mask:
                mask = out_d[1]
            joint_out_split = self.joint_out_fc(dropout, force_recompute=True)

            # This flag means we need to offset labels by + 1 when passing to RNN-T Loss
            # The reason for offset is that we treat logits "A" dimension as [<blank>, valid characters... A-1]
            # Thus, blank-symbol has idx 0 and real symbols must have indices [1:A-1]
            # RNN-T Loss uses labels as indices of logits (in A dimension)
            # The opposite logic must be applied when logits are used for decoder - see transducer_decoder.py
            if self.shift_labels_by_one:
                one = self.builder.aiOnnx.constant(np.array([1]).astype(np.int32))
                targets = self.builder.aiOnnx.add([targets, one])

            compact_log_probs = builder.customOp(
                opName="SparseLogSoftmax",
                opVersion=1,
                domain="com.acme",
                inputs=[joint_out_split, targets, target_lens],
                attributes={},
                numOutputs=1,
            )[0]
            joint_net_output = [compact_log_probs]
            if self.output_dropout_mask:
                joint_net_output.append(mask)
                
        return joint_net_output

    def __build_graph(self, transcription_out, prediction_out, targets, target_lens):

        builder = self.builder
        logger.info("Shapes of Joint-Network Inputs: {}, {}".format(builder.getTensorShape(transcription_out),
                                                                    builder.getTensorShape(prediction_out)))          
        if self.transcription_splitter is not None:
            with self.builder.virtualGraph(0):
                transcription_out_splits = self.transcription_splitter(transcription_out)

            log_probs_compact_splits = []
            if self.output_dropout_mask:
                mask_splits = []
            for split_ind, transcription_out_split in enumerate(transcription_out_splits):
                logger.info("Building compact log probs for split {}".format(split_ind))
                get_log_probs_in = [transcription_out_split, prediction_out, targets, target_lens]
                with self.builder.virtualGraph(0):
                    get_log_probs_out = self.get_compact_log_probs(*get_log_probs_in)
                    log_probs_compact = get_log_probs_out[0]
                    log_probs_compact_splits.append(log_probs_compact)
                    out_idx = 1
                    if self.output_dropout_mask:
                        mask_part = get_log_probs_out[out_idx]
                        out_idx += 1
                        mask_splits.append(mask_part)
                    assert(out_idx == len(get_log_probs_out))

            with self.builder.virtualGraph(0):
                log_probs_compact = builder.aiOnnx.concat(log_probs_compact_splits, axis=1)
                output = [log_probs_compact]
                if self.output_dropout_mask:
                    mask = builder.aiOnnx.concat(mask_splits, axis=1)
                    output.append(mask)
                    
        else:
            
            logger.info("Building compact log probs")
            get_log_probs_in = [transcription_out, prediction_out, targets, target_lens]
            with self.builder.virtualGraph(0):
                get_log_probs_out = self.get_compact_log_probs(*get_log_probs_in)
                log_probs_compact = get_log_probs_out[0]
                output = [log_probs_compact]
                out_idx = 1
                if self.output_dropout_mask:
                    mask = get_log_probs_out[out_idx]
                    out_idx += 1
                    output.append(mask)
                assert(out_idx == len(get_log_probs_out))

        return output


def build_joint_net_model(builder, joint_network, input_values):
    t_cpu, u_cpu, l_cpu, ll_cpu = input_values[0:4]
    t = builder.addInputTensor(popart.TensorInfo(_get_popart_type(t_cpu.dtype.type), t_cpu.shape), "trans")
    u = builder.addInputTensor(popart.TensorInfo(_get_popart_type(u_cpu.dtype.type), u_cpu.shape), "pred")
    l = builder.addInputTensor(popart.TensorInfo(_get_popart_type(l_cpu.dtype.type), l_cpu.shape), "labels")
    ll = builder.addInputTensor(popart.TensorInfo(_get_popart_type(ll_cpu.dtype.type), ll_cpu.shape), "label_lens")
  
    output_dropout_mask = joint_network.output_dropout_mask
    
    w = joint_network.joint_out_fc.weights
    b = joint_network.joint_out_fc.biases
        
    inputs = [t, u, l, ll]
    joint_network_outputs = joint_network(*inputs)
      
    y = joint_network_outputs[0]
    out_idx = 1
    if output_dropout_mask:
        mask = joint_network_outputs[out_idx]
        out_idx += 1
    assert(out_idx == len(joint_network_outputs))

    dt = popart.reservedGradientPrefix() + t
    du = popart.reservedGradientPrefix() + u
    dw = popart.reservedGradientPrefix() + w
    db = popart.reservedGradientPrefix() + b
    outputs = [y, dt, du, dw, db]
    if output_dropout_mask:
        outputs.append(mask)
               
    loss = builder.aiGraphcore.l1loss([y], 1.0, reduction=popart.ReductionType.Mean)
    optimizer = popart.SGD({"defaultLearningRate": (0.1, False)})

    return (inputs, outputs, loss, optimizer)


def joint_net(t_cpu, u_cpu, l_cpu, ll_cpu, splits, dropout_rate, custom_op, custom_dropout, outline, shift_labels_by_one, output_dropout_mask, no_recompute_fc_in, no_recompute_fc_out):   
    logger.debug("\n********** custom: {}, custom dropout: {}, outline: {}, output dropout mask: {}"
                 .format(custom_op, custom_dropout, outline, output_dropout_mask))

    B = t_cpu.shape[0]
    T = t_cpu.shape[1]
    U = u_cpu.shape[2]
    A0 = t_cpu.shape[3]
    A1 = 2
    
    if splits > 0:
        split_size = (T + splits - 1) // splits
    else:
        split_size = 0

    np.random.seed(0)
    builder = popart.Builder()
    joint_network = JointNetwork(builder, T, A0, A1, dropout_rate, np.float32, split_size, custom_op=custom_op, custom_dropout=custom_dropout,
                                 shift_labels_by_one=shift_labels_by_one,
                                 output_dropout_mask=output_dropout_mask, no_recompute_fc_in = no_recompute_fc_in, no_recompute_fc_out = no_recompute_fc_out)   
    input_values = [t_cpu, u_cpu, l_cpu, ll_cpu]
    (inputs, outputs, loss, optimizer) = build_joint_net_model(builder, joint_network, input_values)
    
    logger.debug("outputs = {}".format(outputs))
    
    hw_seed = 0
    output_values = helper_run(builder, inputs, input_values, outputs, 1, True, loss, optimizer, outline, hw_seed)
    return output_values

"""
    Test custom joint net op in different scenarios.
    The test compares the output of custom joint net op and base joint net.
    You can also compare the output of joint net op with custom dropout only.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--t-splits', type=int, default=1,
                        help='Splits along T dimension')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='Dropout_rate')
    parser.add_argument('--outline', action="store_true", default=False,
                        help='Do outlining')
    parser.add_argument('--shift-labels', action="store_true", default=False,
                        help='shift labels by one')
    parser.add_argument('--test1', type=str, choices=["base", "co", "cd"], default="base",
                        help='1st test: base, co=custom joint op, cd=custom dropout only')
    parser.add_argument('--test2', type=str, choices=["base", "co", "cd"], default="base",
                        help='2nd test: base, co=custom joint op, cd=custom dropout only')
    parser.add_argument('--output-dropout-mask', action="store_true", default=False,
                        help='output dropout mask for base op')
    conf = parser.parse_args()
        
    if conf.dropout_rate > 0.0 and conf.test1 != conf.test2:
        print("**********************************************************")
        print("*** Because of non-determinism of popart dropout seed, ***")
        print("*** the test may fail if running 2 different graphs    ***")
        print("*** with non zero dropout mask !                       ***")
        print("*** This is expected.                                  ***")
        print("**********************************************************")

    libc_custom_joint_net = custom_op_utils.load_custom_lib("custom_joint_net")
    custom_op_utils.load_custom_lib("sparse_logsoftmax")
    libc_custom_dropout = custom_op_utils.load_custom_lib("dropout")
    
    fun_joint_net_outputDropoutMask = libc_custom_joint_net.outputDropoutMask
    fun_joint_net_outputDropoutMask.restype = ctypes.c_uint32
    fun_joint_net_outputDropoutMask.argtypes = []
    output_dropout_mask_co = fun_joint_net_outputDropoutMask()
    
    fun_dropout_outputDropoutMask = libc_custom_dropout.outputDropoutMask
    fun_dropout_outputDropoutMask.restype = ctypes.c_uint32
    fun_dropout_outputDropoutMask.argtypes = []
    output_dropout_mask_cd = fun_dropout_outputDropoutMask()
   
    fun_noRecomputeFcIn = libc_custom_joint_net.noRecomputeFcIn
    fun_noRecomputeFcIn.restype = ctypes.c_uint32
    fun_noRecomputeFcIn.argtypes = []
    no_recompute_fc_in = fun_noRecomputeFcIn()
    
    fun_noRecomputeFcOut = libc_custom_joint_net.noRecomputeFcOut
    fun_noRecomputeFcOut.restype = ctypes.c_uint32
    fun_noRecomputeFcOut.argtypes = []
    no_recompute_fc_out = fun_noRecomputeFcOut()
           
    t_cpu = np.array([
            [
                [
                    [1.0, 10.0]
                ],
                [
                    [2.0, 20.0]
                ],
                [
                    [3.0, 30.0]
                ],
                [
                    [4.0, 40.0]
                ]
            ]
        ]).astype(np.float32)
    u_cpu = np.array([
            [
                [
                    [1.0, 0.1],
                    [3.0, 0.3]
                ]
            ]
        ]).astype(np.float32)
    l_cpu = np.array([[0]]).astype(np.int32)
    ll_cpu = np.array([1]).astype(np.int32)

    y_cpus = []
    dt_cpus = []
    du_cpus = []
    dw_cpus = []
    db_cpus = []

    mask_cpus = []
    bwd_mask_cpus = []
    relu_cpus = []
    bwd_relu_cpus = []
        
    test = [conf.test1, conf.test2]

    for i in range(2):
        custom_op = False
        custom_dropout = False
        output_dropout_mask = False
        if test[i] == "co":
            custom_op = True
            output_dropout_mask = output_dropout_mask_co
        elif test[i] == "cd":
            custom_dropout = True
            output_dropout_mask = output_dropout_mask_cd
        elif test[i] == "base":
            if conf.dropout_rate > 0.0:
                output_dropout_mask = conf.output_dropout_mask
        
        output_values = joint_net(t_cpu, u_cpu, l_cpu, ll_cpu, conf.t_splits, conf.dropout_rate, custom_op, custom_dropout, conf.shift_labels, conf.outline,
                                  output_dropout_mask, no_recompute_fc_in, no_recompute_fc_out)
       
        y_cpu, dt_cpu, du_cpu, dw_cpu, db_cpu = output_values[0:5]
        out_idx = 5
        if output_dropout_mask:
            mask_cpu = output_values[out_idx]
            out_idx += 1
        assert(out_idx == len(output_values))

        if output_dropout_mask:
            logger.debug("mask:\n{}".format(mask_cpu.flatten()))
        logger.debug("y:\n{}".format(y_cpu))
        logger.debug("dt:\n{}".format(dt_cpu))
        logger.debug("du:\n{}".format(du_cpu))
        logger.debug("dw:\n{}".format(dw_cpu))
        logger.debug("db:\n{}".format(db_cpu))
                          
        y_cpus.append(y_cpu)
        dt_cpus.append(dt_cpu)
        du_cpus.append(du_cpu)
        dw_cpus.append(dw_cpu)
        db_cpus.append(db_cpu)
        if output_dropout_mask:
            mask_cpus.append(mask_cpu)
        else:
            mask_cpus.append(None)
              
    if mask_cpus[0] is not None and mask_cpus[1] is not None:
        assert(np.array_equal(mask_cpus[0], mask_cpus[1]))
    assert(np.allclose(y_cpus[0], y_cpus[1], atol=1.e-6, rtol=1.e-6))
    
    assert(np.allclose(dt_cpus[0], dt_cpus[1], atol=1.e-6, rtol=1.e-6))
    assert(np.allclose(du_cpus[0], du_cpus[1], atol=1.e-6, rtol=1.e-6))
    assert(np.allclose(dw_cpus[0], dw_cpus[1], atol=1.e-6, rtol=1.e-6))
    assert(np.allclose(db_cpus[0], db_cpus[1], atol=1.e-6, rtol=1.e-6))
    
    logger.info("All test passed")
