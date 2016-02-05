"""Mechanik der Faserverbundwerkstoffe, FEM Code fuer orthotrope, duenne Plattenstrukturen."""
__author__ = "B. Emek Abali"
__license__  = "GNU LGPL Version 3.0 or later"

# FEniCS Pakete bereitstellen:
from fenics import *
import numpy
tic()
class Aufgabe:
    def __init__(self):
        # Optimierung fuer schnelleres Kompilieren:
        parameters["form_compiler"]["cpp_optimize"] = True
    def set_geometry(self,x=500.0,y=50.0,z=20.0,ppl=0.1):
        # Die Geometrie:
        self.xlength = x #in mm
        self.ylength = y #in mm
        self.zlength = z #in mm
        xps          = int(x*ppl)
        yps          = int(y*ppl)
        zps          = int(z*ppl)
        self.mesh    = BoxMesh(Point(0, 0, 0), Point(self.xlength, self.ylength, self.zlength), xps,yps,zps)

    def set_vectorspace(self):
        # Vektorraum fuer Verchiebungen mit linearen Elementen:
        self.V = VectorFunctionSpace(self.mesh, 'CG', 1)
        
    def set_boundaries(self):
        # Berandung:
        self.boundaries = FacetFunction('size_t',self.mesh)
        self.domains    = CellFunction('size_t',self.mesh)
        self.left       = CompiledSubDomain('near(x[0], 0) && on_boundary')
        self.right      = CompiledSubDomain('near(x[0], length) && on_boundary',length = self.xlength)
        
#       self.top   = CompiledSubDomain('near(x[2], zl) && (x[0]-a)*(x[0]-a) + (x[1]-b)*(x[1]-b) < r*r && on_boundary', zl=zlength, a=200.0, b=10.0, r=20.0)
        self.top    = CompiledSubDomain('near(x[2], zl) && x[0] < a && x[0] > b && on_boundary', zl=self.zlength, a=200.0, b=150.0)

        # Zuerst alle als 0 markieren, dann die rechte Berandung als 
        self.boundaries.set_all(0)
        # Beim Integrieren wird ds(1) ein Flaechenintegral ueber die obere Berandung fuer die aufgebrachte Kraft bedeuten
        self.top.mark(self.boundaries, 1)
        self.dA     = Measure('ds')[self.boundaries]
        self.V      = Measure('dx')[self.domains]
        self.hat_t  = Expression(('0.0','0.0','A'),A=20.0) #in MPa
        #self.hat_t = Expression(('0.0','0.0','A*x[1]/yl'),A=150.0,yl=ylength)

    def set_dirichlet(self):
        # Dirichlet Randbedingungen:
        self.null = Constant((0.0,0.0,0.0))
        self.bc1  = DirichletBC(self.V, self.null, self.left)
        self.bc2  = DirichletBC(self.V, self.null, self.right)
        self.bc   = [self.bc1,self.bc2]
    def trad_form(self):
        # Die variationelle Formulierung:
        self.u     = TrialFunction(V)
        self.del_u = TestFunction(V)

    def VoigtToTensor(self,A):
        A11, A12, A13, A14, A15, A16 = A[0,0], A[0,1], A[0,2], A[0,3], A[0,4], A[0,5]
        A22, A23, A24, A25, A26 = A[1,1], A[1,2], A[1,3], A[1,4], A[1,5]
        A33, A34, A35, A36 = A[2,2], A[2,3], A[2,4], A[2,5]
        A44, A45, A46 = A[3,3], A[3,4], A[3,5]
        A55, A56 = A[4,4], A[4,5]
        A66 = A[5,5]
        A21, A31, A41, A51, A61 = A12, A13, A14, A15, A16
        A32, A42, A52, A62 = A23, A24, A25, A26
        A43, A53, A63 = A34, A35, A36
        A54, A64 = A45, A46
        A65 = A56
        return as_tensor([ \
        [ \
        [ [A11,A16,A15], [A16,A12,A14], [A15,A14,A13]] , \
        [ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
        [ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] \
        ] , [ \
        [ [A61,A66,A65], [A66,A62,A64], [A65,A64,A63]] , \
        [ [A21,A26,A25], [A26,A22,A24], [A25,A24,A23]] , \
        [ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] \
        ] , [ \
        [ [A51,A56,A55], [A56,A52,A54], [A55,A54,A53]] , \
        [ [A41,A46,A45], [A46,A42,A44], [A45,A44,A43]] , \
        [ [A31,A36,A35], [A36,A32,A34], [A35,A34,A33]] ] \
        ])
    def set_isotropy(self,nu=0.3,E=210000.0 ):
        # Isotrop:

        self.nu = nu
        self.E  = E
        self.G  = self.E/(2.0*(1.0+self.nu))

        self.S_voigt = numpy.array([ \
        [1./self.E, -self.nu/self.E, -self.nu/self.E, 0, 0, 0],\
        [-self.nu/self.E, 1./self.E, -self.nu/self.E, 0, 0, 0],\
        [-self.nu/self.E, -self.nu/self.E, 1./self.E, 0, 0, 0],\
        [0, 0, 0, 1./self.G, 0, 0],\
        [0, 0, 0, 0, 1./self.G, 0],\
        [0, 0, 0, 0, 0, 1/self.G]  ])

        self.C_voigt = numpy.linalg.inv(S_voigt)
        self.C       = VoigtToTensor(C_voigt)
    def set_Kronecker_delta(self,dim=3,indices=4):
        # Kronecker delta in 3D
        self.delta                  = Identity(dim)
        self.i,self.j,self.k,self.l = indices(indices)
        
    def set_Dehnungstensor(self):
        # Dehnungstensor:
        self.eps = as_tensor(1.0/2.0*(self.u[self.i].dx(self.j)+self.u[self.j].dx(self.i)) , (self.i,self.j))
        
    def set_Cauchy_Spannungstensor(self):
        # Cauchy Spannungstensor:
        self.sigma = as_tensor(self.C[self.i,self.j,self.k,self.l]*self.eps[self.k,self.l] , (self.i,self.j))

    def set_goal_functions(self):
        self.a = self.sigma[self.j,self.i]*self.del_u[self.i].dx(sefl.j)*sefl.dV
        self.L = self.hat_t[self.i]*self.del_u[self.i]*self.dA(1)

    def solve(self):
        self.disp    = Function(self.V)
        solve(self.a == self.L , sefl.disp, self.bc)
    def output(self):
        file = File('Verschiebungen.pvd')
        file << self.disp
        
    def F_nach_Tsai_Hill(self):
        # Berechnung von F nach Tsai-Hill
        self.eps_ = as_tensor(1.0/2.0*(self.disp[self.i].dx(self.j)+self.disp[self.j].dx(self.i)) , (self.i,self.j))
        self.s_   = as_tensor(self.C[self.i,self.j,self.k,self.l]*self.eps_[self.k,self.l] , (self.i,self.j))
        
    def set_L_Zugfestigkeit(self,sL=320.0,sT=320.0,tau=55.0 ):#MPa
        # L-Zugfestigkeit          = T-Zugfestigkeit (Orthotropie), Schubfestigkeit
        self.sL, self.sT, self.tau = sL,sT,tau
        self.F_                    = as_tensor((s_[0,0]/sL)**2 + (s_[1,1]/sT)**2 - s_[0,0]*s_[1,1]/sT**2 + (s_[0,1]/tau)**2, ())
        self.F                     = project(F_, FunctionSpace(mesh, 'CG', 1))
    def output2(self):
        file = File('Tsai_Hill_F.pvd')
        file << self.F
        print 'Fertig in ',toc(),' Sekunden!'

    def sigma_epsilon_plot(self):
        #sigma/epsilon plot
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as pylab
        pylab.rc('text', usetex=True )
        pylab.rc('font', family='serif', serif='cm', size=25 )
        pylab.rc('legend', fontsize=25)
        pylab.rc(('xtick.major','ytick.major'), pad=15)

        fig = pylab.figure(1, figsize=(12,8))
        fig.clf()
        pylab.subplots_adjust(bottom=0.18)
        pylab.subplots_adjust(left=0.18)
        pylab.xlabel(r'$\varepsilon_{33}$ in $\%$')
        pylab.ylabel(r'$\sigma_{33}$ in MPa')
        pylab.grid(True)
        hat_t = Expression(('0.0','0.0','A'),A=0)
        stress_plot,strain_plot = [0],[0]
        P = Point(xlength/2., ylength/2., zlength/2.)
        for tau in numpy.linspace(0.,1.,5):
            hat_t.A = 2000.*tau
            L = hat_t[i]*del_u[i]*dA(1)
            solve(a == L , disp, bc)
            stress_value = project(s_,TensorFunctionSpace(mesh,'CG',1))(P)[8]
            strain_value = project(eps_,TensorFunctionSpace(mesh,'CG',1))(P)[8]
            print stress_value, strain_value
            stress_plot.append( stress_value )
            strain_plot.append( strain_value*100. )
            pylab.plot(strain_plot, stress_plot,'ro-',markersize=6, linewidth=3)
            pylab.savefig('SpannungsDehnungsDiagramm.pdf')
            
    def execute_Beispiel(self):
        """simply replicates the execution order of the script"""
        self.set_geometry()
        self.set_vectorspace()
        self.set_boundaries()
        self.set_dirichlet()
        self.trad_form()
        self.VoigtToTensor()
        self.set_isotropy()
        self.set_Kronecker_delta()
        self.set_Dehnungstensor()
        self.set_Cauchy_Spannungstensor()
        self.set_goal_functions()
        self.solve()
        self.output()
        self.F_nach_Tsai_Hill()
        self.set_L_Zugfestigkeit()
        self.output2()
        self.sigma_epsilon_plot()
        
        #self.
        
if __name__=="__main__":
    Beispiel=Aufgabe()
    Beispiel.execute_Beispiel()

