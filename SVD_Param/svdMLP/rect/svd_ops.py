import numpy
import theano.tensor
from theano.tensor import as_tensor_variable
from theano.gof import Op, Apply

def Hprod(h, u, k):
    alpha = 2*numpy.dot(h[ -k:],u[-k:])
    h_out = h.copy()
    h_out[ -k:] -= alpha * u[-k:]
    return h_out

def Hgrad(h, u, g, k):
    alpha = 2*numpy.dot(h[-k:], u[-k:])
    beta = 2*numpy.dot(g[-k:], u[-k:])
    u_bar = -alpha*g - beta*h + alpha*beta*u
    g_out = g.copy()
    g_out -=  beta*u
    return g_out, u_bar

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
        m_u = U.shape[1]; n_u = U.shape[0]
        m_v = V.shape[1]; n_v = V.shape[0]
        n_sig = Sig.shape[0]

        hv = numpy.zeros_like(V)
        hu = numpy.zeros_like(U)
        hsig = numpy.zeros_like(Sig)

        # Compute V'*h = H[N-n+1]H[N-n]...H[N]h
        hv[:,0] = Hprod(h, V[:,0], n_v)
        for m in range(1, m_v):
            hv[:,m] = Hprod(hv[:,m-1], V[:,m], n_v - m)
        h1 = hv[0 : n_sig, m_v-1] # h1.shape = (n_sig,)

        # Compute Sig*h1
        hsig = numpy.multiply(Sig, h1)

        # Compute U*h = H[N]H[N-1]...H[N-n+1]
        hu[:,m_u-1] = Hprod(hsig, U[:,m_u-1], n_u-m_u+1)
        for k in range(m_u-2, -1,-1):
            hu[:,k] = Hprod(hu[:,k+1], U[:, k], n_u - k)
        c = hu[0 : n_sig, 0]
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
    Calculating U_bar and h_bar.
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
        m_u = U.shape[1]; n_u = U.shape[0]
        m_v = V.shape[1]; n_v = V.shape[0]
        n_sig = Sig.shape[0]
        hv = numpy.zeros_like(V)
        hu = numpy.zeros_like(U)
        hsig = numpy.zeros_like(Sig)
        n = U.shape[1]

        #Compute V'*h = H[N-n+1]H[N-n]...H[N]h
        hv[:,0] = Hprod(h, V[:,0], n_v)
        for m in range(1, m_v):
            hv[:,m] = Hprod(hv[:,m-1], V[:,m], n_v - m)
        h1 = hv[0 : n_sig, m_v-1] # h1.shape = (n_sig,)

        # Compute Sig*h1
        hsig = numpy.multiply(Sig, h1)

        # Compute U*h = H[N]H[N-1]...H[N-n+1]
        hu[:,m_u-1] = Hprod(hsig, U[:,m_u-1], n_u-m_u+1)
        for m in range(m_u-2, 0,-1): # no need to compute the last one
            hu[:,m] = Hprod(hu[:,m+1], U[:, m], n_u - m)

        # Start computing gradients
        U_bar = numpy.zeros_like(U)
        V_bar = numpy.zeros_like(V)
        Sig_bar = numpy.zeros_like(Sig)
        g = c_bar
        h_bar = numpy.zeros_like(h)
        '''
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
        g = c_bar - numpy.dot(U, c_tilde)
        '''
        # Compute U_bar
        for m in range(m_u-1):
            g, U_bar[:, m] = Hgrad(hu[:, m+1], U[:,m], g, n_u-m)
        g, U_bar[:, m_u-1] = Hgrad(hsig, U[:, m_u-1], g, n_u-m_u+1)

        # Compute Sig_bar
        Sig_bar = numpy.multiply(g, h1)
        g = numpy.multiply(g, Sig)
        g = numpy.append(g, numpy.zeros(n_v-n_sig))
        # Compute V_bar
        for k in range(m_v-1, 0, -1):
            g, V_bar[:,k] = Hgrad(hv[:,k-1], V[:,k], g, n_v - k )
        g, V_bar[:, 0] = Hgrad( h, V[:,0], g, n_v)

        # Output results
        outputs[0][0] = U_bar
        outputs[1][0] = V_bar
        outputs[2][0] = Sig_bar
        outputs[3][0] = g

    def infer_shape(self, node, shapes):
        return [shapes[0], shapes[1], shapes[2], shapes[3]]


