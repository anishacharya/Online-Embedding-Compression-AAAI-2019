import numpy
import theano.tensor
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply


class svd_H_WY(Op):
    """
    Calculating  U*Sig*V'*h
    """
    __props__ = ()

    def make_node(self, U, V, Sig, h):
        U = as_tensor_variable(U)
        V = as_tensor_variable(V)
        Sig = as_tensor_variable(Sig)
        h = as_tensor_variable(h)
        assert U.ndim == 2
        assert V.ndim == 2
        assert Sig.ndim == 1
        assert h.ndim == 1
        out_dtype = theano.scalar.upcast(U.dtype,V.dtype, Sig.dtype, h.dtype)
        c = theano.tensor.vector(dtype=out_dtype)
        return Apply(self, [U, V, Sig, h], [c])

    def perform(self, node, inputs, outputs):
        U, V, Sig, h = inputs
        n = U.shape[1]

        # Compute V'*h = H[N-n+1]H[N-n]...H[N]h
        h_hat = numpy.dot(V.T, h)
        tmp = numpy.zeros_like(h)
        c_tilde = numpy.zeros(n)
        for k in range(0, n):
            c_tilde[ k ] = 2. * (h_hat[ k ] - V[ k:, k ].dot(tmp[ k: ]))
            tmp[ k: ] = c_tilde[ k ] * V[ k:, k ] + tmp[ k: ]
        h1 = h - numpy.dot(V, c_tilde)

        # Compute Sig*h1
        h2 = numpy.multiply(Sig, h1)

        # Compute U*h = H[N]H[N-1]...H[N-n+1]
        h_hat = numpy.dot(U.T, h2)
        tmp = numpy.zeros_like(h)
        c_tilde = numpy.zeros(n)
        for k in range(1, n + 1):
            c_tilde[n - k] = 2. * (h_hat[n - k] - U[n - k:, n - k].dot(tmp[n - k:]))
            tmp[n - k:] = c_tilde[n - k] * U[n - k:, n - k] + tmp[n - k:]
        c = h2 - numpy.dot(U, c_tilde)

        outputs[0][0] = c

    def infer_shape(self, node, shapes):
        return [shapes[3]]

    def grad(self, inputs, output_gradients):
        U, V, Sig, h = inputs
        c_bar = output_gradients[0]
        return svd_H_WYGrad()(U, V, Sig, h, c_bar)


svd_H_wy = svd_H_WY()

class svd_H_WYGrad(Op):
    """
    Calculating U_bar V_bar Sigma_bar and h_bar.
    """
    __props__ = ()

    def make_node(self, U, V, Sig, h, c_bar):
        U = as_tensor_variable(U)
        V = as_tensor_variable(V)
        Sig = as_tensor_variable(Sig)
        h = as_tensor_variable(h)
        assert U.ndim == 2
        assert V.ndim == 2
        assert Sig.ndim == 1
        assert h.ndim == 1
        assert c_bar.ndim == 1
        out_dtype = theano.scalar.upcast(U.dtype, V.dtype, Sig.dtype, h.dtype, c_bar.dtype)
        U_bar = theano.tensor.matrix(dtype=out_dtype)
        V_bar = theano.tensor.matrix(dtype=out_dtype)
        Sig_bar = theano.tensor.vector(dtype=out_dtype)
        h_bar = theano.tensor.vector(dtype=out_dtype)
        return Apply(self, [U, V, Sig, h, c_bar], [U_bar, V_bar, Sig_bar, h_bar])

    def perform(self, node, inputs, outputs):
        U, V, Sig, h, c_bar = inputs
        n = U.shape[1]

        # First compute h1 = V'*h
        h_hat = numpy.dot(V.T, h)
        tmp0 = numpy.zeros_like(h)
        h1_tilde = numpy.zeros(n)
        for k in range(0, n):
            h1_tilde[ k ] = 2. * (h_hat[ k ] - V[ k:, k ].dot(tmp0[ k: ]))
            tmp0[ k: ] = h1_tilde[ k ] * V[ k:, k ] + tmp0[ k: ]
        h1 = h - numpy.dot(V, h1_tilde)

	    # Compute  h2 = Sig*h1
        h2 = numpy.multiply(Sig, h1)

        # Start computing gradients
        U_bar = numpy.zeros_like(U)
        V_bar = numpy.zeros_like(V)
        Sig_bar = numpy.zeros_like(Sig)

        # Compute U_bar
        h_tilde = numpy.zeros(n)
        c_tilde = numpy.zeros(n)
        tmp1 = numpy.zeros_like(h)
        tmp2 = numpy.zeros_like(h)

        h_hat = numpy.dot(U.T, h2)
        c_hat = numpy.dot(U.T, c_bar)
        for k in range(n):
            # Compute c_tilde;  h_tilde = T^{-1} U^T h
            h_tilde[n-k-1] = 2. * (h_hat[n-k-1] - U[n-k-1:, n-k-1].dot(tmp1[n-k-1:]))
            tmp1[n-k-1:] += h_tilde[n-k-1] * U[n-k-1:, n-k-1]
            # Compute h_bar;  c_tilde = T^{-T} U^T c_bar
            c_tilde[k] = 2. * (c_hat[k] - U[k:, k].dot(tmp2[k:]))
            tmp2[k:] += c_tilde[k] * U[k:, k]

        tmp1 *= 0
        tmp2 *= 0
        for k in range(n - 1):
            tmp1[k:] += U[k:, k] * c_tilde[k]
            U_bar[:, k+1] += tmp1 * h_tilde[k+1]
            tmp2[n-k-1:] += U[n-k-1:, n-k-1] * h_tilde[n-k-1]
            U_bar[:, n-k-1] += tmp2 * c_tilde[n-k-1]
        tmp2 += U[:, 0] * h_tilde[0]
        U_bar[:, 0] += tmp2 * c_tilde[0]

        U_bar -= numpy.outer(c_bar, h_tilde) + numpy.outer(h2, c_tilde)
        h2_bar = c_bar - numpy.dot(U, c_tilde)

        # Compute Sig_bar
        Sig_bar = numpy.multiply(h2_bar, h1)
        h1_bar = numpy.multiply(h2_bar, Sig)
        # Compute V_bar
        h_tilde *= 0
        c_tilde *= 0
        tmp1 *= 0 # -H
        tmp2 *= 0 # -g

        h_hat = numpy.dot(V.T, h)
        c_hat = numpy.dot(V.T, h1_bar)
        for k in range(n):
            # Compute c_tilde;  h_tilde = T^{-1} U^T h
            h_tilde[k] = 2. * (h_hat[k] - V[k:, k].dot(tmp1[k:]))
            tmp1[k:] += h_tilde[k] * V[k:, k]
            # Compute h_bar;  c_tilde = T^{-T} U^T c_bar
            c_tilde[n-k-1] = 2. * (c_hat[n-k-1] - V[n-k-1:, n-k-1].dot(tmp2[n-k-1:]))
            tmp2[n-k-1:] += c_tilde[n-k-1] * V[n-k-1:, n-k-1]

        tmp1 *= 0 # -g
        tmp2 *= 0 # -H
        for k in range(n - 1):
            tmp1[n-k-1:] += V[n-k-1:, n-k-1] * c_tilde[n-k-1]
            V_bar[:, n-k-2] += tmp1 * h_tilde[n-k-2]
            tmp2[k:] += V[k:, k] * h_tilde[k]
            V_bar[:, k] += tmp2 * c_tilde[k]
        tmp2[n-1:] += V[n-1:, n-1] * h_tilde[n-1]
        V_bar[:, n-1] += tmp2 * c_tilde[n-1]

        V_bar -= numpy.outer(h1_bar, h_tilde) + numpy.outer(h, c_tilde)
        h_bar = h1_bar - numpy.dot(V, c_tilde)

        # Output results
        outputs[0][0] = U_bar
        outputs[1][0] = V_bar
        outputs[2][0] = Sig_bar
        outputs[3][0] = h_bar

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1], shapes[2], shapes[3]]
