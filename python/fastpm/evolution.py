from __future__ import print_function
import numpy
import logging

from abopt.vmad import VM

from pmesh.pm import ParticleMesh, RealField

import fastpm.operators as operators

class Evolution(VM):
    def __init__(self, pm, B=1, shift=0, dtype='f8'):
        self.pm = pm
        self.fpm = ParticleMesh(Nmesh=pm.Nmesh * B, BoxSize=pm.BoxSize, dtype=dtype)
        self.q = operators.create_grid(self.pm, shift=shift, dtype=dtype)
        VM.__init__(self)

    @VM.microcode(aout=['b'], ain=['a'])
    def copy(self, a):
        if hasattr(a, 'copy'):
            return a.copy()
        else:
            return 1.0 * a

    @VM.microcode(ain=['dlin_k'], aout=['prior'])
    def Prior(self, dlin_k, powerspectrum):
        return dlin_k.cdot(dlin_k,
                    metric=lambda k: 1 / (powerspectrum(k) / dlin_k.BoxSize.prod()) ** 0.5,
                    independent=False)
    @Prior.grad
    def _(self, dlin_k, powerspectrum, _prior):
        w = dlin_k.cdot_gradient(_prior, 
                    metric=lambda k: 1 / (powerspecrum(k) / dlin_k.BoxSize.prod()) ** 0.5,
                    independent=False)
        w[...] *= 2 # because this is self cdot.
        return w

    @VM.microcode(aout=['s', 'p'], ain=['dlin_k'])
    def Displace(self, dlin_k, D1, v1, D2, v2):
        q = self.q
        dx1 = operators.lpt1(dlin_k, q)
        source = operators.lpt2source(dlin_k)
        dx2 = operators.lpt1(source, q)
        s = D1 * dx1 + D2 * dx2
        p = v1 * dx1 + v2 * dx2
        return s, p

    @Displace.grad
    def _(self, dlin_k, _s, _p, D1, v1, D2, v2):
        q = self.q
        grad_dx1 = _p * v1 + _s * D1
        grad_dx2 = _p * v2 + _s * D2

        if grad_dx1 is not VM.Zero:
            gradient = operators.lpt1_gradient(dlin_k.pm, q, grad_dx1)
        else:
            gradient = VM.Zero

        if grad_dx2 is not VM.Zero:
            gradient_lpt2source = operators.lpt1_gradient(dlin_k.pm, q, grad_dx2)
            gradient[...] +=  operators.lpt2source_gradient(dlin_k, gradient_lpt2source)

        return gradient

    @VM.microcode(aout=['meshforce'], ain=['mesh'])
    def MeshForce(self, mesh, d, factor):
        deltak = field.r2c(out=Ellipsis)
        f = deltak.apply(laplace_kernel) \
                  .apply(diff_kernel(d), out=Ellipsis) \
                  .c2r(out=Ellipsis)
        f[...] *= factor
        return f

    @MeshForce.grad
    def _(self, _meshforce, d, factor):
        _mesh = _meshforce.c2r_gradient()\
                           .apply(laplace_kernel, out=Ellipsis) \
                           .apply(diff_kernel(d), out=Ellipsis) \
                           .r2c_gradient(out=Ellipsis)
        _mesh[...] *= factor
        return _mesh

    @VM.microcode(ain=['f', 'meshforce', 's'], aout=['f'])
    def Readout(self, s, meshforce, d, f):
        if f is VM.Zero:
            f = numpy.empty_like(self.q)
        x = s + self.q
        layout = self.fpm.decompose(x)
        meshforce.readout(x, layout=layout, out=f[:, d])
        return f

    @VM.microcode(aout=['f'], ain=['s'])
    def Force(self, s, factor):
        density_factor = self.fpm.Nmesh.prod() / self.fpm.Nmesh.prod()
        x = s + self.q
        return operators.gravity(x, self.fpm, factor=density_factor * factor, f=None)

    @Force.grad
    def _(self, s, _f, factor):
        density_factor = self.fpm.Nmesh.prod() / self.pm.Nmesh.prod()

        if _f is VM.Zero:
            return VM.Zero
        else:
            x = s + self.q
            return operators.gravity_gradient(x, self.pm, density_factor * factor, _f)

    @VM.microcode(aout=['mesh'], ain=['s'])
    def Paint(self, s):
        x = s + self.q
        mesh = self.pm.create(mode='real')
        layout = self.pm.decompose(x)
        mesh.paint(x, layout=layout, hold=False)
        # to have 1 + \delta on the mesh
        mesh[...] *= mesh.pm.Nmesh.prod() / self.pm.Nmesh.prod()
        return mesh

    @Paint.grad
    def _(self, _mesh, s):
        if _mesh is VM.Zero:
            return VM.Zero
        else:
            x = s + self.q
            layout = _mesh.pm.decompose(x)
            _s, junk = _mesh.paint_gradient(x, layout=layout, out_mass=False)
            _s[...] *= _mesh.pm.Nmesh.prod() / self.pm.Nmesh.prod()
            return _s

    @VM.microcode(aout=['mesh'], ain=['mesh'])
    def Diff(self, mesh, data_x, sigma_x):
        diff = mesh + -1 * data_x
        diff[...] /= sigma_x[...]
        return diff

    @Diff.grad
    def _(self, _mesh, sigma_x):
        _mesh[...] /= sigma_x
        return _mesh

    @VM.microcode(aout=['chi2'], ain=['variable'])
    def Chi2(self, variable):
        variable = variable * variable
        if isinstance(variable, RealField):
            return variable.csum()
        else:
            return self.pm.comm.allreduce(variable.sum(dtype='f8'))

    @Chi2.grad
    def _(self, _chi2, variable):
        return variable * (2 * _chi2)

    @VM.microcode(aout=['p'], ain=['f', 'p'])
    def Kick(self, f, p, dda):
        p[...] += f * dda
        return p

    @Kick.grad
    def _(self, _p, dda):
        if _p is VM.Zero:
            return VM.Zero, VM.Zero
        return _p * dda, _p

    @VM.microcode(aout=['s'], ain=['p', 's'])
    def Drift(self, p, s, dyyy):
        s[...] += p * dyyy
        return s

    @Drift.grad
    def _(self, _s, dyyy):
        if _s is VM.Zero:
            return VM.Zero, VM.Zero
        return _s * dyyy, _s

class LPT(Evolution):
    def __init__(self, pm, shift=0):
        Evolution.__init__(self, pm, B=1, shift=shift)

    def simulation(self, pt, aend, order):
        code = Evolution.code(self)
        if order == 1:
            code.Displace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=0,
                          v2=0,
                         )
        if order == 2:
            code.Displace(D1=pt.D1(aend), 
                          v1=pt.f1(aend) * pt.D1(aend) * aend ** 2 * pt.E(aend),
                          D2=pt.D2(aend),
                          v2=pt.f2(aend) * pt.D2(aend) * aend ** 2 * pt.E(aend),
                         )
        return code

class KickDriftKick(Evolution):
    def __init__(self, pm, B=1, shift=0):
        Evolution.__init__(self, pm, B=B, shift=shift)

    def simulation(self, pt, astart, aend, Nsteps):
        code = Evolution.code(self)
        code.Displace(D1=pt.D1(astart), 
                      v1=pt.f1(astart) * pt.D1(astart) * astart ** 2 * pt.E(astart),
                      D2=pt.D2(astart),
                      v2=pt.f2(astart) * pt.D2(astart) * astart ** 2 * pt.E(astart),
                     )
        code.Force(factor=1.5 * pt.Om0)

        a = numpy.linspace(astart, aend, Nsteps + 1, endpoint=True)
        def K(ai, af, ar):
            return 1 / (ar ** 2 * pt.E(ar)) * (pt.Gf(af) - pt.Gf(ai)) / pt.gf(ar)
        def D(ai, af, ar):
            return 1 / (ar ** 3 * pt.E(ar)) * (pt.Gp(af) - pt.Gp(ai)) / pt.gp(ar)

        for ai, af in zip(a[:-1], a[1:]):
            ac = (ai * af) ** 0.5

            code.Kick(dda=K(ai, ac, ai))
            code.Drift(dyyy=D(ai, ac, ac))
            code.Drift(dyyy=D(ac, af, ac))
            code.Force(factor=1.5 * pt.Om0)
            code.Kick(dda=K(ac, af, af))

        return code

