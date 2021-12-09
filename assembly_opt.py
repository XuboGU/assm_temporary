import numpy as np
import os
import time
import pickle

from neorl import DE
import openmc

start = time.time()

# os.environ['OPENMC_CROSS_SECTIONS'] = '/home/adt/neorl/openmc/endfb71_hdf5/cross_sections.xml'

def fuel_universe(width=100/11, z=0.5):
    ''' Create triso fuel pin universe
    # ref:https://docs.openmc.org/en/stable/examples/triso.html?highlight=TRISO#Modeling-TRISO-Particles
    '''
    ## triso fuel material
    fuel = openmc.Material(name='Fuel') # fuel
    fuel.set_density('g/cm3', 10.5)
    fuel.add_nuclide('U235', 4.6716e-02)
    fuel.add_nuclide('U238', 2.8697e-01)
    fuel.add_nuclide('O16',  5.0000e-01)
    fuel.add_nuclide('C0', 1.6667e-01)

    buff = openmc.Material(name='Buffer') # carbon as buffer
    buff.set_density('g/cm3', 1.0)
    buff.add_nuclide('C0', 1.0)
    buff.add_s_alpha_beta('c_Graphite') # bound atom cross section at thermal energies

    PyC1 = openmc.Material(name='PyC1') # pyrolytic carbon
    PyC1.set_density('g/cm3', 1.9)
    PyC1.add_nuclide('C0', 1.0)
    PyC1.add_s_alpha_beta('c_Graphite')

    PyC2 = openmc.Material(name='PyC2') # pyrolytic carbon
    PyC2.set_density('g/cm3', 1.87)
    PyC2.add_nuclide('C0', 1.0)
    PyC2.add_s_alpha_beta('c_Graphite')

    SiC = openmc.Material(name='SiC') # SiC shell
    SiC.set_density('g/cm3', 3.2)
    SiC.add_nuclide('C0', 0.5)
    SiC.add_element('Si', 0.5)

    # moderator material: YH_1.7
    YH2 = openmc.Material()  # moderator
    YH2.set_density('g/cm3', 4.28)
    YH2.add_element('Y', 1/2.7)
    YH2.add_element('H',1.7/2.7)

    ## Create TRISO universe
    id0 = 100000
    spheres = [openmc.Sphere(r=1e-4*r, surface_id=id0+i)
            for i,r in enumerate([215., 315., 350., 385.])]
    cells = [openmc.Cell(fill=fuel, region=-spheres[0], cell_id=id0+1),
            openmc.Cell(fill=buff, region=+spheres[0] & -spheres[1], cell_id=id0+2),
            openmc.Cell(fill=PyC1, region=+spheres[1] & -spheres[2], cell_id=id0+3),
            openmc.Cell(fill=SiC, region=+spheres[2] & -spheres[3], cell_id=id0+4),
            openmc.Cell(fill=PyC2, region=+spheres[3], cell_id=id0+5)]

    triso_univ = openmc.Universe(universe_id=100001, name='Triso particle', cells=cells)

    # pack the TRISO particles in a 100/11 cm x 100/11 cm x 1 cm box centered at the origin
    min_x = openmc.XPlane(x0=-width/2, surface_id=200001)
    max_x = openmc.XPlane(x0=width/2, surface_id=200002)
    min_y = openmc.YPlane(y0=-width/2, surface_id=200003)
    max_y = openmc.YPlane(y0=width/2, surface_id= 200004)
    min_z = openmc.ZPlane(z0=-z, surface_id=200005)
    max_z = openmc.ZPlane(z0=z, surface_id=200006)
    region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z

    # randomly select locations for the TRISO particles
    outer_radius = 425.*1e-4
    centers = openmc.model.pack_spheres(radius=outer_radius, region=region, pf=0.3) # create centers for triso to pack in
    trisos = [openmc.model.TRISO(outer_radius, triso_univ, center) for center in centers] # create trisos at centers

    box = openmc.Cell(cell_id=200001, region=region)
    lower_left, upper_right = box.region.bounding_box
    shape = (1, 1, 1)
    pitch = (upper_right - lower_left)/shape
    lattice = openmc.model.create_triso_lattice( # pack trisos in the box with moderator YH2
        trisos, lower_left, pitch, shape, YH2)
    box.fill = lattice      

    # create TRISO pin universe
    universe = openmc.Universe(universe_id=200001, name='Triso pin', cells=[box])

    return universe

def assembly(pitch=100, z=0.5, void_loc_x=np.array([]), void_loc_y=np.array([]), xyboundary_type='vacuum', zboundary_type='reflective',fuel_pin_universe=''):
    """Create a 11X11 assembly model.

    This model is a reflected 11x11 fuel assembly from. Note that the 
    number of particles/batches is initially set very low for testing purposes.

    """

    model = openmc.model.Model()

    # Create boundary planes to surround the geometry
    pitch = pitch # in `cm``
    min_x = openmc.XPlane(x0=-pitch/2, boundary_type=xyboundary_type)
    max_x = openmc.XPlane(x0=+pitch/2, boundary_type=xyboundary_type)
    min_y = openmc.YPlane(y0=-pitch/2, boundary_type=xyboundary_type)
    max_y = openmc.YPlane(y0=+pitch/2, boundary_type=xyboundary_type)
    min_z = openmc.ZPlane(z0=-z, boundary_type=zboundary_type)
    max_z = openmc.ZPlane(z0=+z, boundary_type=zboundary_type)

    # void pin surfaces
    void_pitch = pitch/11
    min_void_x = openmc.XPlane(x0=-void_pitch/2) # boundary type: none
    max_void_x = openmc.XPlane(x0=+void_pitch/2) 
    min_void_y = openmc.YPlane(y0=-void_pitch/2)
    max_void_y = openmc.YPlane(y0=+void_pitch/2)
    min_void_z = openmc.ZPlane(z0=-z)
    max_void_z = openmc.ZPlane(z0=+z)

    # Create a void universe
    void_universe = openmc.Universe(universe_id=900001, name='void')
    void_cell = openmc.Cell(name='void', region=+min_void_x & -max_void_x & \
        +min_void_y & -max_void_y & +min_void_z & -max_void_z)
    void_universe.add_cell(void_cell)

    # Create fuel assembly Lattice
    assembly = openmc.RectLattice(lattice_id=700001, name='Fuel Assembly') # provide lattice_id to avoid confliction with triso lattice
    assembly.pitch = (pitch/11, pitch/11)  
    assembly.lower_left = (-pitch/2, -pitch/2) 

    # Create array indices for void locations in lattice
    template_x = void_loc_x
    template_y = void_loc_y

    # Create 11x11 array of universes 
    # print('-debug: fuel_pin_universe', fuel_pin_universe, '\ntype:', type(fuel_pin_universe))
    assembly.universes = np.tile(fuel_pin_universe, (11, 11))
    assembly.universes[template_x, template_y] = void_universe
    # print('-debug: assembly univ:', assembly.universes)

    # Create root Cell
    root_cell = openmc.Cell(cell_id=500001, name='root cell', fill=assembly)
    root_cell.region = +min_x & -max_x & +min_y & -max_y & +min_z & -max_z

    # Create root Universe
    model.geometry.root_universe = openmc.Universe(universe_id=600001, name='root universe')
    model.geometry.root_universe.add_cell(root_cell)

    # geometry
    geometry=openmc.Geometry(model.geometry.root_universe)
    geometry.export_to_xml()

    # materials 
    all_materials = geometry.get_all_materials()
    materials = openmc.Materials(all_materials.values())
    materials.export_to_xml()

    # simulation parameters
    model.settings.batches = 3 # total number of batches to execute
    model.settings.inactive = 2 # inactive batches used in a k-eigenvalue calculation
    model.settings.particles = 15 # number of neutrons to simulate
    # space: patial distribution of source sites; \
    # A “box” spatial distribution has coordinates sampled uniformly in a parallelepiped
    model.settings.source = openmc.Source(space=openmc.stats.Box(
        [-pitch/2, -pitch/2, -z], [pitch/2, pitch/2, z], only_fissionable=True))
    model.settings.export_to_xml()
    model.settings.output = {'tallies': False, 'summary': False}
    model.settings.sourcepoint_write = False
    # plot 
    # plot = openmc.Plot.from_geometry(geometry, basis='xy')
    # plot.color_by = 'material'
    # plot.to_ipython_image()

    # plots
        # assm = openmc.Plot(name='assembly')
        # assm.basis = 'xy'
        # assm.origin = (0.0, 0.0, 0)
        # assm.width = (pitch, pitch)
        # assm.pixels = (pitch*50, pitch*50)
        # assm.filename = 'smr_assm_xy_0'
        # assm.color_by = 'material'
        # model.plots.append(assm)

        # assm2 = openmc.Plot(name='assembly2')
        # assm2.basis = 'xy'
        # assm2.origin = (0.0, 0.0, z-0.1)
        # assm2.width = (pitch, pitch)
        # assm2.pixels = (pitch*50, pitch*50)
        # assm2.filename = 'smr_assm_xy_0d49'
        # assm2.color_by = 'material'
        # model.plots.append(assm2)

        # assm3 = openmc.Plot(name='assembly3')
        # assm3.basis = 'xz'
        # assm3.origin = (0.0, 0.0, 0)
        # assm3.width = (pitch, z)
        # assm3.pixels = (pitch*50, int(z*50))
        # assm3.filename = 'smr_assm_xz_0'
        # assm3.color_by = 'material'
        # model.plots.append(assm3)

        # model.plots.export_to_xml()

    return model

# # triso_pin_univ = fuel_universe()
# # print(triso_pin_univ)
# # print('triso_pin_universe type:', type(triso_pin_univ))

# time_p = time.time() - start
# print('time for creating triso univ:', time_p)

# # save triso univ class 
# # output_triso = open("triso.pkl", 'wb')
# # str = pickle.dumps(triso_pin_univ)
# # output_triso.write(str)
# # print('-*- triso univ. class saved! -*- ')
# # output_triso.close()

# # # load triso univ. class
# with open('triso.pkl','rb') as file:
#     triso_pin_univ = pickle.loads(file.read())

# assem_model = assembly(fuel_pin_universe=triso_pin_univ)
# print('model:', assem_model)
# # assem_model.run()
# time_p2 = time.time() - start
# print('time for creating assembly:', time_p2 - time_p)

# # run model
# assem_model.run()

# time_p3 = time.time() - start
# print('Running OpenMC Time:', time_p3 - time_p2)


## call NEORL to find the optimal geometry config to max k-eff ## 
# Define the fitness function
with open('triso.pkl','rb') as file:
    triso_pin_univ = pickle.loads(file.read())

def FIT(arr):

    print('type arr is:', type(arr)) # list
    print('arr:', arr) 

    total_pin = 121 # number of pin in assembly
    fuel_limit = 57 # limit fuel units

    list_x, list_y = [], [] # store locations of void pin
    for idx, val in enumerate(arr):
        row, col = idx//11, idx%11 # the location (row, col) in the assembly
        if val>(57/121): # void pin probability
            list_x.append(row)
            list_y.append(col)

    lx = np.array(list_x)
    ly = np.array(list_y)
    model = assembly(void_loc_x=lx, void_loc_y=ly, fuel_pin_universe=triso_pin_univ)
    result_r = model.run(output=True, threads=16)
    sp = openmc.StatePoint(result_r)
    k_combined = sp.k_combined
    k_combined_nom = k_combined.nominal_value
    k_combined_stddev = k_combined.std_dev

    # penalty of over-use fuel
    penalty = -1e5
    used_fuel = total_pin - len(lx)
    if used_fuel > fuel_limit: return_val = k_combined_nom + penalty
    else: return_val = k_combined_nom

    return return_val

# Setup the parameter space(enrichment of U belongs to[0,4.0])
nx=121
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', 0, 1]

# use DE to find the optimal U enrichment
de=DE(mode='max', bounds=BOUNDS, fit=FIT, npop=5, F=0.5, CR=0.3,  ncores=1, seed=100)
x_best, y_best, de_hist=de.evolute(ngen=5, verbose=1)
print('---DE Results---', )
print('x:', x_best)
print('y:', y_best)
print('DE History:\n', de_hist)
end = time.time()
running_time = end - start
print('running time:\n', running_time)


# use JAYA to find the optimal U enrichment
    # jaya=JAYA(mode='min', bounds=BOUNDS, fit=FIT, npop=10, ncores=1, seed=100)
    # x_best, y_best, jaya_hist=jaya.evolute(ngen=30, verbose=1)
    # print('---JAYA Results---', )
    # print('x:', x_best)
    # print('y:', y_best)
    # print('JAYA History:\n', jaya_hist)
    # end = time.time()
    # running_time = end - start
    # print('running time:\n', running_time)