import numpy as np
import theano as th
T = th.tensor
import theano.misc.pycuda_init
from pycuda.compiler import SourceModule
import theano.sandbox.cuda as cuda

#max thread per block for CUDA
CUDA_MAX_TPB = 512

with open('kernels/simple_conv.cu','r') as f:
    mod = SourceModule(
        f.read())
    _fn_sconv_row = mod.get_function('sconv_packed_fwd_row_f32_ker')
    _fn_sconv_col = mod.get_function('sconv_packed_fwd_col_f32_ker')
    _fn_sconv_row_bwd = mod.get_function('sconv_packed_bwd_row_f32_ker')
    _fn_sconv_col_bwd = mod.get_function('sconv_packed_bwd_col_f32_ker')

class SimpleConvGradOp(th.Op):
    '''This Op is supposed to be used by SimpleConvOp.grad()'''
    def infer_shape(self, node, shapes_):
        dedy_shape_ = shapes_[0]
        assert len(dedy_shape_)==4
        bsize, c, w, h = dedy_shape_
        dedx_shape = (bsize, c//5, w,h)
        tmp_shape = (bsize, (c*2)//5, w,h)
        return [dedx_shape, tmp_shape]

    def make_node(self, s_dedy_):
        assert s_dedy_.dtype == 'float32'
        s_dedy = cuda.basic_ops.gpu_contiguous(
            cuda.basic_ops.as_cuda_ndarray_variable(s_dedy_))
        return th.Apply(self, [s_dedy], [s_dedy.type(), s_dedy.type()])

    def __call__(self, *inp_):
        '''only return first output from make_node -> the 2nd output is temporal buffer'''
        return self.make_node(*inp_).outputs[0]

    def make_thunk(self, node, storage_map, compute_map, recycling, impl=None):
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            pv_dedy = inputs[0]
            pv_dedx = outputs[0]
            pv_tmp = outputs[1]

            ishape = list(pv_dedy[0].shape)
            if len(ishape)!=4:
                raise ValueError('Expect 4D tensor input, got "%s"'%ishape)
            bsize, c, w,h = ishape
            if c%5 != 0:
                raise ValueError('Number of channels in gradient is supposed to be times of 5')
            if w>CUDA_MAX_TPB or h>CUDA_MAX_TPB: raise ValueError('For now, image width/height must not exceed 512')
            c//=5
            oshape = (bsize,c,w,h)
            tshape = (bsize,c*2,w,h)

            if pv_dedx[0] is not None:
                if oshape != tuple(pv_dedx[0].shape):
                    pv_dedx[0] = cuda.CudaNdarray.zeros(oshape)
            else:
                pv_dedx[0] = cuda.CudaNdarray.zeros(oshape)

            if pv_tmp[0] is not None:
                if tshape != tuple(pv_tmp[0].shape):
                    pv_tmp[0] = cuda.CudaNdarray.zeros(tshape)
            else:
                pv_tmp[0] = cuda.CudaNdarray.zeros(tshape)

            if not pv_dedx[0].is_c_contiguous():
                raise RuntimeError('Array is not c-contiguous')
            if not pv_dedy[0].is_c_contiguous():
                raise RuntimeError('Array is not c-contiguous')
            if not pv_tmp[0].is_c_contiguous():
                raise RuntimeError('Array is not c-contiguous')

            blk_x = CUDA_MAX_TPB//h
            blk_z = 1
            if blk_x>w:
                blk_x = w
                blk_z = CUDA_MAX_TPB//(w*h)
            blk = (blk_x, h, blk_z)
            grd = ((w+blk_x-1)//blk_x, 1, bsize*c//blk_z)
            smem_bytes = (h+2)*blk_x*16*blk_z
            conf1 = dict(block=blk, grid=grd, shared=smem_bytes)

            blk_y = CUDA_MAX_TPB//w
            blk_z = 1
            if blk_y>h:
                blk_y = h
                blk_z = CUDA_MAX_TPB//(w*h)
            blk = (w, blk_y, blk_z)
            grd = (1, (h+blk_y-1)//blk_y, bsize*c//blk_z)
            smem_bytes = (w+2)*blk_y*8*blk_z
            conf2 = dict(block=blk, grid=grd, shared=smem_bytes)
            _fn_sconv_col_bwd(
                pv_dedx[0], pv_tmp[0], pv_dedy[0], np.intc(w), np.intc(h), np.intc(bsize*c),
                **conf1
            )
            _fn_sconv_row_bwd(
                pv_dedx[0], pv_tmp[0], np.intc(w), np.intc(h), np.intc(bsize*c),
                **conf2
            )

        return thunk
class SimpleConvOp(theano.Op):
    __props__ = ()
    def make_node(self, s_x_):
        assert s_x_.dtype == 'float32'
        s_x = cuda.basic_ops.gpu_contiguous(
            cuda.basic_ops.as_cuda_ndarray_variable(s_x_))
        return th.Apply(self, [s_x], [s_x.type()])

    def make_thunk(self, node, storage_map, compute_map, recycling, impl=None):
        global _fn_sconv_row, _fn_sconv_col
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        def thunk():
            pv_x = inputs[0]
            pv_y = outputs[0]
            ishape = list(pv_x[0].shape)
            if len(ishape)!=4:
                raise ValueError('Expect 4D tensor input, got "%s"'%ishape)
            bsize, c, w,h = ishape
            if w>CUDA_MAX_TPB or h>CUDA_MAX_TPB: raise ValueError('For now, image width/height must not exceed %s'%CUDA_MAX_TPB)
            oshape = ishape.copy()
            oshape[1] *= 5
            if pv_y[0] is not None:
                if oshape != list(pv_y[0].shape):
                    pv_y[0] = cuda.CudaNdarray.zeros(oshape)
            else:
                pv_y[0] = cuda.CudaNdarray.zeros(oshape)
                #FIXME: doesn't seems guarantee c-contiguous allocation
            if not pv_y[0].is_c_contiguous():
                raise RuntimeError('Array is not c-contiguous')
            if not pv_x[0].is_c_contiguous():
                raise RuntimeError('Array is not c-contiguous')

            blk_y = CUDA_MAX_TPB//w
            blk_z = 1
            if blk_y>h:
                blk_y = h
                blk_z = CUDA_MAX_TPB//(w*h)
            blk = (w, blk_y, blk_z)
            grd = (1, (h+blk_y-1)//blk_y, bsize*c//blk_z)
            smem_bytes = (w+2)*blk_y*4*blk_z
            conf1 = dict(block=blk, grid=grd, shared=smem_bytes)

            blk_x = 512//h
            blk_z = 1
            if blk_x>w:
                blk_x = w
                blk_z = 512//(w*h)
            blk = (blk_x, h, blk_z)
            grd = ((w+blk_x-1)//blk_x, 1, bsize*c//blk_z)
            smem_bytes = (h+2)*blk_x*8*blk_z
            conf2 = dict(block=blk, grid=grd, shared=smem_bytes)
            _fn_sconv_row(
                pv_y[0], pv_x[0], np.intc(w), np.intc(h), np.intc(bsize*c),
                **conf1
            )
            _fn_sconv_col(
                pv_y[0], pv_x[0], np.intc(w), np.intc(h), np.intc(bsize*c),
                **conf2
            )

        return thunk

    def grad(self, inp_li_, out_li_):
        assert len(inp_li_)==1
        assert len(out_li_)==1
        s_dedy = out_li_[0]
        return [SimpleConvGradOp()(s_dedy)]

    def infer_shape(self, node_, input_shapes_):
        assert len(input_shapes_) == 1
        ishape = list(input_shapes_[0])
        assert len(ishape)==4
        ishape[1] *= 5
        return [tuple(ishape)]

simple_conv = SimpleConvOp()
