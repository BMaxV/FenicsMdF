"""Mechanik der Faserverbundwerkstoffe, FEM Code fuer orthotrope, duenne Plattenstrukturen."""
__author__  = "B. Emek Abali"
__license__ = "GNU LGPL Version 3.0 or later"

# FEniCS Pakete bereitstellen:
from fenics import *
import numpy
tic()
# Optimierung fuer schnelleres Kompilieren:
parameters["form_compiler"]["cpp_optimize"] = True
# Die Geometrie:
xlength = 200.0 #in mm
ylength = 200.0 #in mm
zlength = 5.0 #in mm
mesh    = BoxMesh(Point(0, 0, 0), Point(xlength, ylength, zlength), 40, 40, 5)
# Vektorraum fuer Verchiebungen mit linearen Elementen:
V = VectorFunctionSpace(mesh, 'CG', 1)
# Berandung:
boundaries = FacetFunction('size_t',mesh)
domains    = CellFunction('size_t',mesh)
left       = CompiledSubDomain('near(x[0], 0) && on_boundary')
right      = CompiledSubDomain('near(x[0], length) && on_boundary',length    = xlength)
#top       = CompiledSubDomain('near(x[2], zl) && (x[0]-a)*(x[0]-a) + (x[1]-b)*(x[1]-b) < r*r && on_boundary', zl=zlength, a=200.0, b=10.0, r=20.0)
top        = CompiledSubDomain('near(x[2], zl) && x[0] < a && x[0] > b && on_boundary', zl=zlength, a=110.0, b=90.0)

# Zuerst alle als 0 markieren, dann die rechte Berandung als 
boundaries.set_all(0)
# Beim Integrieren wird ds(1) ein Flaechenintegral ueber die obere Berandung fuer die aufgebrachte Kraft bedeuten
top.mark(boundaries, 1)
dA     = Measure('ds')[boundaries]
dV     = Measure('dx')[domains]
hat_t  = Expression(('0.0','0.0','A'),A=20.0) #in MPa
#hat_t = Expression(('0.0','0.0','A*x[1]/yl'),A=150.0,yl=ylength)

# Dirichlet Randbedingungen:
null = Constant((0.0,0.0,0.0))
bc1  = DirichletBC(V, null, left)
bc2  = DirichletBC(V, null, right)
bc   = [bc1,bc2]

# Die variationelle Formulierung:
u = TrialFunction(V)
del_u = TestFunction(V)

def VoigtToTensor(A):
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

# Isotrop:
nu = 0.3
E  = 210000.0 #MPa
G  = E/(2.0*(1.0+nu))

S_voigt = numpy.array([ \
[1./E, -nu/E, -nu/E, 0, 0, 0],\
[-nu/E, 1./E, -nu/E, 0, 0, 0],\
[-nu/E, -nu/E, 1./E, 0, 0, 0],\
[0, 0, 0, 1./G, 0, 0],\
[0, 0, 0, 0, 1./G, 0],\
[0, 0, 0, 0, 0, 1/G]  ])

C_voigt = numpy.linalg.inv(S_voigt)
C = VoigtToTensor(C_voigt)
# Kronecker delta in 3D
delta = Identity(3)
i,j,k,l = indices(4)
# Dehnungstensor:
eps = as_tensor(1.0/2.0*(u[i].dx(j)+u[j].dx(i)) , (i,j))
# Cauchy Spannungstensor:
sigma = as_tensor(C[i,j,k,l]*eps[k,l] , (i,j))

a = sigma[j,i]*del_u[i].dx(j)*dV
L = hat_t[i]*del_u[i]*dA(1)

disp = Function(V)
solve(a == L , disp, bc)

file = File('Verschiebungen.pvd')
file << disp

# Berechnung von F nach Tsai-Hill
eps_ = as_tensor(1.0/2.0*(disp[i].dx(j)+disp[j].dx(i)) , (i,j))
s_ = as_tensor(C[i,j,k,l]*eps_[k,l] , (i,j))
# L-Zugfestigkeit = T-Zugfestigkeit (Orthotropie), Schubfestigkeit 
sL, sT, tau = 320.0, 320.0, 55.0 #MPa
F_ = as_tensor((s_[0,0]/sL)**2 + (s_[1,1]/sT)**2 - s_[0,0]*s_[1,1]/sT**2 + (s_[0,1]/tau)**2, ())
F = project(F_, FunctionSpace(mesh, 'CG', 1))

file = File('Tsai_Hill_F.pvd')
file << F
print 'Fertig in ',toc(),' Sekunden!'

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
hat_t                   = Expression(('0.0','0.0','A'),A=0)
stress_plot,strain_plot = [0],[0]
P                       = Point(xlength/2., ylength/2., zlength/2.)
for tau in numpy.linspace(0.,1.,5):
    hat_t.A      = 2000.*tau
    L            = hat_t[i]*del_u[i]*dA(1)
    solve(a      == L , disp, bc)
    stress_value = project(s_,TensorFunctionSpace(mesh,'CG',1))(P)[8]
    strain_value = project(eps_,TensorFunctionSpace(mesh,'CG',1))(P)[8]
    print stress_value, strain_value
    stress_plot.append( stress_value )
    strain_plot.append( strain_value*100. )
    pylab.plot(strain_plot, stress_plot,'ro-',markersize=6, linewidth=3)
    pylab.savefig('SpannungsDehnungsDiagramm.pdf')

