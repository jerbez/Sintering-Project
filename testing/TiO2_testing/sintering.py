#!/usr/bin/env python

'''
Sintering simulation via Ostwald Ripening of metalic clusters deposited on the surface 
based on NVT ensemble and metropolis moves.

Borna Zandkarimi 2020

Revised by Jake Erbez 2026
'''

import numpy as np
import timeit, math, copy
import param
from scipy.special import expit          # for handling very small exp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import ScalarFormatter

# Define distance function 
def distance(x1, y1, x2, y2):
    r = ((x1 - x2)**2.0 + (y1 - y2)**2.0)**(0.5)
    return r

# Define function to locate center of mass between two clusters for more realistic collision
def calcCOM(OUTPUT_data,i,j):
    xi = OUTPUT_data[i][2]
    yi = OUTPUT_data[i][3]
    mi = OUTPUT_data[i][0]
    xj = OUTPUT_data[j][2]
    yj = OUTPUT_data[j][3]
    mj = OUTPUT_data[j][0]

    Lx = param.maxx
    Ly = param.maxy

    dx = xj - xi
    dx -= round(dx / Lx) * Lx
    xj_img = xi + dx

    xcm = (mi * xi + mj * xj_img) / (mi + mj)
    xcm %= Lx

    dy = yj - yi
    dy -= round(dy / Ly) * Ly
    yj_img = yi + dy

    ycm = (mi * yi + mj * yj_img) / (mi + mj)
    ycm %= Ly

    return xcm, ycm

# Function to check for periodic images of clusters overlapping
# Inputs: coords: OUTPUT_DATA (data of all clusters); current_x: x position of cluster being evaluated;
#         current_y: y position of cluster being evaluated; current_radius: radius of cluster being evaluated;
#         current_LCG: largest cluster currently

def boundaryOverlapCheck(coords, current_x, current_y, current_radius, current_LCG ):
    overlap = False
    targetPosition = -1

   #print("maxx: " + str(param.maxx) + " maxy: " + str(param.maxy))
    # check whether is the beginning
    if current_LCG <= 0:
        return overlap, targetPosition, current_LCG

    # check whether this LCG is smaller than current radius
    if current_LCG <= current_radius :
        current_LCG = current_radius
    
    isRight = False
    isLeft = False
    isUp = False
    isDown = False

    # determine which kinds overlap is happening here
    overlapPotentialType = []

    # right most case, left most, up case, down case
    if current_x + current_radius >=  param.maxx:
        isRight = True
        overlapPotentialType.append( "right")
    if current_x - current_radius <= param.minx:
        isLeft = True
        overlapPotentialType.append("left" )
    if current_y + current_radius>= param.maxy:
        isUp = True
        overlapPotentialType.append("up")
    if current_y - current_radius<= param.miny:
        isDown = True
        overlapPotentialType.append("down")
    
    # no boundary overlap case
    if overlapPotentialType == [] :
        return overlap, targetPosition, current_LCG
    
    # newClusterList[][0] = x
    # newClusterList[][1] = y
    # newClusterList[][2] = radius
    # newClusterList[][3] = types
    # newClusterList[][4] = position
    newClusterList = []
    
    #print(overlapPotentialType)
    for types in overlapPotentialType:
        if types == "right" :
            # delta is the longest length that other potential overlap cluster can exist
            delta = current_x + current_radius - param.maxx + 2 * current_LCG
            #counting position
            n = 0
            for cluster in coords:
                if cluster[2] <=  param.minx + delta:
                    newClusterList.append([cluster[2] + param.maxx, cluster[3], cluster[1], cluster[0], n])       
                n += 1
        if types == "left" :
            # this delt should be a negative number
            delta = current_x - current_radius + param.minx - 2 * current_LCG
            n = 0
            for cluster in coords:
                if cluster[2] >= param.maxx + delta:
                    newClusterList.append([cluster[2] - param.maxx, cluster[3], cluster[1], cluster[0], n])
                n += 1
        if types == "up" :
            delta = current_y + current_radius - param.maxy + 2 * current_LCG
            n = 0
            for cluster in coords:
                if cluster[3] <= param.miny + delta:
                    newClusterList.append([cluster[2], cluster[3] +param.maxy, cluster[1], cluster[0], n])
                n += 1
        if types == "down" :
            delta = current_y - current_radius + param.miny - 2 * current_LCG
            n = 0
            for cluster in coords:
                if cluster[3] >= param.maxy +delta:
                    newClusterList.append([cluster[2], cluster[3] - param.maxy, cluster[1],cluster[0], n])
                n += 1
    
    # check for corner cases
    if isRight and isUp :
        n = 0
        for cluster in coords:
                if (cluster[2] <= current_LCG) and (cluster[3] <= current_LCG):
                    newClusterList.append([cluster[2] + param.maxx, cluster[3] + param.maxy, cluster[1], cluster[0], n])
                n += 1
    if isRight and isDown :
        n = 0
        for cluster in coords:
                if (cluster[2] <= current_LCG) and (cluster[3] >= param.maxy - current_LCG):
                    newClusterList.append([cluster[2] + param.maxx, cluster[3] - param.maxy, cluster[1], cluster[0], n])
                n += 1
    if isLeft and isUp :
        n = 0
        for cluster in coords:
                if (cluster[2] >= param.maxx - current_LCG) and (cluster[3] <= current_LCG):
                    newClusterList.append([cluster[2] - param.maxx, cluster[3] + param.maxy, cluster[1], cluster[0], n])
                n += 1
    if isLeft and isDown :
        n = 0
        for cluster in coords:
                if (cluster[2] >= param.maxx - current_LCG) and (cluster[3] >= param.maxy - current_LCG):
                    newClusterList.append([cluster[2] - param.maxx, cluster[3] - param.maxy, cluster[1], cluster[0], n])
                n += 1
    
    OVERLAPNUMBERCOUNT = 0
    # checking overlap
    for testCluster in newClusterList:
        #  print("radius: " + str(current_radius))
        #  print("radius: " + str(testCluster[2]))
        if distance(current_x,current_y, testCluster[0],testCluster[1]) <= current_radius + testCluster[2]:
            #print(str(current_x) + " " + str(current_y) + " " +str( testCluster[0]) + " " + str(testCluster[1]) + " " + str(current_radius) + " " + str(testCluster[2]))
            targetPosition = testCluster[4]
            overlap = True
            OVERLAPNUMBERCOUNT += 1 
            break
               
    #print(overlap)
    # print(len(newClusterList))
    #  print(OVERLAPNUMBERCOUNT) 
    # return whether overlap, position of overlap cluster
    return overlap, targetPosition, current_LCG

# Function that checks for overlapping clusters. If any are found, they are combined.
def overlap_check(Clusters, OUTPUT_data, LCG):
    ovlp = False
    idx = []
    idx_pair = []

    # Loop over all clusters and check for overlap
    for i in range(len(OUTPUT_data) - 1):
        if (ovlp == True):
            break
        else:
            for j in range(i+1, len(OUTPUT_data)):
                
                # Check if the distance between two clusters is less than the sum of their radii
                R1 = OUTPUT_data[i][1] 
                R2 = OUTPUT_data[j][1] 
                d  = distance(OUTPUT_data[i][2],OUTPUT_data[i][3],OUTPUT_data[j][2],OUTPUT_data[j][3])
                if (d < R1 + R2):
                   # print("normal overlap")
                    #print(f'd:{d}, R1:{R1}, R2:{R2}')
                    ovlp = True
                    idx.append(i)
                    idx.append(j)
                    idx_pair.append([i,j])
                    #print(f'idx_pair:{idx_pair}')
                    #print(f'OUTPUT_data[i]:{OUTPUT_data[i]}')
                    #print(f'OUTPUT_data[j]:{OUTPUT_data[j]}')
                    #merge
                    numnew = OUTPUT_data[i][0] + OUTPUT_data[j][0]
                    if numnew > 8 :
                        for k in range(len(Clusters)):
                            if (Clusters[k][0] == numnew):
                                Rnew = Clusters[k][1]
                                Enew = Clusters[k][4]
                                if LCG < Rnew:
                                    LCG = Rnew
                    else:
                        Rnew, Enew = boltzmannPopulationForNewCluster(numnew)
                        if LCG < Rnew:
                            LCG = Rnew

                    xnew, ynew = calcCOM(OUTPUT_data,i,j)
                    #print(f'COM:{xnew,ynew}')
                    OUTPUT_data.append([numnew,Rnew,xnew,ynew,Enew])
                    break

            # overlap boundary check
            ovlp1 = False
            ovlp1, overlapPosition, current_LCG = boundaryOverlapCheck(OUTPUT_data, OUTPUT_data[i][2],  OUTPUT_data[i][3], OUTPUT_data[i][1], LCG)
            if ovlp1 == True:
                ovlp = True
                LCG = current_LCG
                j1 = overlapPosition
                
                # eliminate the possiblity that overlap itself
                if (i == j1):
                    ovlp = False
                    break

                idx.append(i)
                idx.append(j1)
                idx_pair.append([i,j1])
                #print(f'maxx:{param.maxx},maxy:{param.maxy}')
                #print(f'idx_pair:{idx_pair}')
                #print(f'OUTPUT_data[i]:{OUTPUT_data[i]}')
                #print(f'OUTPUT_data[j]:{OUTPUT_data[j]}')
                #print("boundary overlap")
                #print("LCG: " + str(LCG))
                #print("x1: " + str(OUTPUT_data[i][2]) + " y1: " + str(OUTPUT_data[i][3])+ " r1: " + str(OUTPUT_data[i][1]) + " x2: " + str(OUTPUT_data[j1][2]) + " y2: " + str(OUTPUT_data[j1][3])+ " r2: " + str(OUTPUT_data[j1][1]))
                #merge
                numnew = OUTPUT_data[i][0] + OUTPUT_data[j1][0]
                if numnew > 8:
                    for k in range(len(Clusters)):    
                        if (Clusters[k][0] == numnew):
                            Rnew = Clusters[k][1]
                            Enew = Clusters[k][4]
                            if LCG < Rnew:
                                LCG = Rnew
                else:
                    Rnew, Enew = boltzmannPopulationForNewCluster(numnew)
                    if LCG < Rnew:
                        LCG = Rnew

                xnew, ynew = calcCOM(OUTPUT_data,i,j)
                #print(f'COM:{xnew,ynew}')
                OUTPUT_data.append([numnew,Rnew,xnew,ynew,Enew])
                break

    # Return data with the position and atomic number of the new cluster
    if (ovlp == True):
        if (idx[0]<idx[1]):
            OUTPUT_data.pop(idx[0]) # remove i 
            OUTPUT_data.pop(idx[1]-1) # remove j -1
        elif (idx[0]>idx[1]):
            OUTPUT_data.pop(idx[0]) # remove i 
            OUTPUT_data.pop(idx[1]) # remove j  
    return OUTPUT_data, ovlp, idx_pair, LCG

def PES_finder(X_new, Y_new, PES_copy, N_mesh):
    finder = False
    for ii in range(N_mesh): 
        dx =  np.abs( param.xstep_max*( math.ceil(X_new/param.xstep_max) % N_meshx ) - PES_copy[ii][1] )
        dy =  np.abs( param.ystep_max*( math.ceil(Y_new/param.ystep_max) % N_meshy ) - PES_copy[ii][2] )
        if ( dx  < 0.1 and dy  < 0.1 ):    
            # PES found 
            E_PES  = PES_copy[ii][3] 
            finder = True
            break    
    if (finder == False):
        raise ValueError('Could not find PES for single atom. Bad move! Check you initial setup')      
    return E_PES

# Function that searches for cluster energies and radii from the DATA file. The inputs are the index of a specific cluster and
# whether a monomer is to be added or removed from that cluster.
def Cluster_finder(Clusters, OUTPUT_data, irmv, iadd, which='both'):
    cf_start = timeit.default_timer()
    P_add = []
    P_rmv = []
    E_add = []
    E_rmv = []
    R_add = []
    R_rmv = []

    # Find data of the new cluster size for either the 'add' or the 'remove' case. For sizes 1-8, include ensemble probabilities.
    for j in range(Ncluster_tot):     
        if (Clusters[j][0] == OUTPUT_data[iadd][0] + 1 and (which == 'add' or which == 'both') ) :
            E_add.append(Clusters[j][4])
            R_add.append(Clusters[j][1])
            P_add.append(expit(-beta*(Clusters[j][4])))
        
        if (Clusters[j][0] == OUTPUT_data[irmv][0] - 1 and (which == 'remove' or which == 'both') ) :
            E_rmv.append(Clusters[j][4])
            R_rmv.append(Clusters[j][1])
            P_rmv.append(expit(-beta*(Clusters[j][4])))

    # Select a cluster isomer to add based on the boltzman probability and back calculate the energy from the probability
    if ( which == 'add' or which == 'both' ): 
        P_add_norm = [icount / sum(P_add) for icount in P_add]
        cluster_idx_add = np.random.choice(np.arange(len(P_add_norm)), 1, p=P_add_norm, replace=False)[0] 
        Enew_add = E_add[cluster_idx_add]
        Rnew_add = R_add[cluster_idx_add]

    # Select a cluster isomer to remove based on the boltzman probability and back calculate the energy from the probability
    if ( which == 'remove' or which == 'both' ): 
        P_rmv_norm = [icount / sum(P_rmv) for icount in P_rmv]
        cluster_idx_rmv = np.random.choice(np.arange(len(P_rmv_norm)), 1, p=P_rmv_norm, replace=False)[0]
        Enew_rmv = E_rmv[cluster_idx_rmv]
        Rnew_rmv = R_rmv[cluster_idx_rmv]

    cf_end = timeit.default_timer()

    print(cf_end - cf_start)

    # Return new energy and radius
    if (which == 'both'):
        return Enew_add, Enew_rmv, Rnew_add, Rnew_rmv
    elif (which == 'add'):
        return Enew_add, Rnew_add
    elif (which == 'remove'):
        return Enew_rmv, Rnew_rmv
 
# READ INPUT   
PES           = [] # potential energy surface element, x, y, z, E
INIT_data     = [] # initial cluster R, x, y, E
Clusters      = [] # all possible cluster R and E              
beta          = param.beta 
hartree_to_ev = param.hartree_to_ev   
Ncluster = len(open('INIT').readlines(  )) - 1 # Number of clusters on the surface during simulation which can be changed
Ncluster_tot = len(open('DATA').readlines(  )) - 1 # Total number of unique different clusters
Metro_Max     = param.MMAX   # num of Metropolis steps
write_step    = param.wstep  # writing metropolis each wstep
N_mesh        = param.N_mesh # total number of PES points
N_meshx       = N_meshy = (N_mesh)**(0.5) # number of mesh points in x and y direction
numprimcell   = param.numprimecellFactor*param.num_clust + param.num_single_atom
xdups = ydups = (int(numprimcell**0.5) + 2)

with open('INIT','r') as f1:
    f1.readline()
    data = f1.readlines()
    for line in data:  
        INIT_data.append([ int(line.split()[0]), 
                           float(line.split()[1]), 
                           float(line.split()[2]), 
                           float(line.split()[3]),
                           float(line.split()[4]) ])

with open('DATA','r') as f2:
    f2.readline()
    data = f2.readlines()
    for line in data:  
        Clusters.append([ int(line.split()[0]), 
                          float(line.split()[1]), 
                          float(line.split()[2]), 
                          float(line.split()[3]),
                          float(line.split()[4]) ])

with open('PES','r') as f3:
    data = f3.readlines()
    for line in data:   
        PES.append([ str(line.split()[0]), 
                     float(line.split()[1]), 
                     float(line.split()[2]), 
                     float(line.split()[3]) ])

# Boltzmann population information, specific to TiO2
T=700
hartree_to_ev = 27.2114
k_J           = 1.38064852e-23
k_ev          = 8.6173303e-5
k_hartree     = k_ev/hartree_to_ev
kT            = k_hartree*T
beta          = 1.0/kT

Pt2=[1.09778751,1.09970170,1.10494276,1.10750419,1.10750419]
Pt2_index=[0,1,2,3,4]
Pt2_boltz_weight=[math.exp(-Pt2[i]/kT) for i in range(0,len(Pt2))]
Q2=sum(Pt2_boltz_weight)
Pt2_prob=[Pt2_boltz_weight[i]/Q2 for i in range(0,len(Pt2_boltz_weight))]
Pt2_data=[[3.0218605,1.09778751],[3.0283540,1.09970170],[3.0056904 ,1.10494276],[2.9766112 ,1.10750419],
          [2.8958918 ,1.10750419]]

Pt3=[0.90056183,0.91199504,0.91510903]
Pt3_index=[0,1,2]
Pt3_boltz_weight=[math.exp(-Pt3[i]/kT) for i in range(0,len(Pt3))]
Q3=sum(Pt3_boltz_weight)
Pt3_prob=[Pt3_boltz_weight[i]/Q3 for i in range(0,len(Pt3_boltz_weight))]
Pt3_data=[[3.0040290,0.90056183],[2.9851504,0.91199504],[2.8275859,0.91510903]]

Pt4=[0.72210192,0.72929952,0.73153563,0.73305648,0.73571588,0.73606268]
Pt4_index=[0,1,2,3,4,5]
Pt4_boltz_weight=[math.exp(-Pt4[i]/kT) for i in range(0,len(Pt4))]
Q4=sum(Pt4_boltz_weight)
Pt4_prob=[Pt4_boltz_weight[i]/Q4 for i in range(0,len(Pt4_boltz_weight))]
Pt4_data=[[ 3.6684145,0.72210192],[3.0623059,0.72929952],[3.6433934,0.73153563],[3.2190594,0.73305648],
         [3.0076024,0.73571588],[3.0904500,0.73606268]]

Pt5=[0.5417517,0.54831806]
Pt5_index=[0,1]
Pt5_boltz_weight=[math.exp(-Pt5[i]/kT) for i in range(0,len(Pt5))]
Q5=sum(Pt5_boltz_weight)
Pt5_prob=[Pt5_boltz_weight[i]/Q5 for i in range(0,len(Pt5_boltz_weight))]
Pt5_data=[[3.7748643 ,0.54175170],[3.6812070,0.54831806]]

Pt6=[0.36589531,0.37585195,0.37773049,0.37906140,0.38048965]
Pt6_index=[0,1,2,3,4]
Pt6_boltz_weight=[math.exp(-Pt6[i]/kT) for i in range(0,len(Pt6))]
Q6=sum(Pt6_boltz_weight)
Pt6_prob=[Pt6_boltz_weight[i]/Q6 for i in range(0,len(Pt6_boltz_weight))]
Pt6_data=[[4.1423044, 0.36589531],[4.2300866 ,0.37585195],[4.1883873 ,0.37773049],[3.9624653,0.37906140],
     [4.1217715,0.38048965]]

Pt7=[ 0.17865857,0.18249841,0.18304948,0.18388096,0.18630240,0.19132231,0.19230703]
Pt7_index=[0,1,2,3,4,5,6]
Pt7_boltz_weight=[math.exp(-Pt7[i]/kT) for i in range(0,len(Pt7))]
Q7=sum(Pt7_boltz_weight)
Pt7_prob=[Pt7_boltz_weight[i]/Q7 for i in range(0,len(Pt7_boltz_weight))]
Pt7_data=[ [4.2377875,0.17865857],[4.2400853,0.18249841],[4.9129259,0.18304948],[4.7864910,0.18388096],
          [4.4981330,0.18630240],[4.1891448,0.19132231],[4.9218588,0.19230703]]

Pt8=[0.00000000,0.00707852,0.01104738,0.01222103,0.01320434]
Pt8_index=[0,1,2,3,4]
Pt8_boltz_weight=[math.exp(-Pt8[i]/kT) for i in range(0,len(Pt8))]
Q8=sum(Pt8_boltz_weight)
Pt8_prob=[Pt8_boltz_weight[i]/Q8 for i in range(0,len(Pt8_boltz_weight))]
Pt8_data=[ [4.6229707,0.00000000],[4.5883157 ,0.00707852],[4.6106749,0.01104738],[4.2228037,0.01222103],
          [4.4044703,0.01320434]]

# For the case of pt2 to pt8, assign boltzman weighted random radii and energies
def boltzmannPopulationForNewCluster(numberOfAtoms) :
    assignedRadius = 0
    assignedEnergy = 0
    if numberOfAtoms==2:
        index = np.random.choice(Pt2_index,p=Pt2_prob)
        assignedRadius = Pt2_data[index][0]
        assignedEnergy = Pt2_data[index][1]
        return assignedRadius, assignedEnergy
    if numberOfAtoms==3:
        index = np.random.choice(Pt3_index,p=Pt3_prob)
        assignedRadius = Pt3_data[index][0]
        assignedEnergy = Pt3_data[index][1]
        return assignedRadius, assignedEnergy
    if numberOfAtoms==4:
        index = np.random.choice(Pt4_index,p=Pt4_prob)
        assignedRadius = Pt4_data[index][0]
        assignedEnergy = Pt4_data[index][1]
        return assignedRadius, assignedEnergy
    if numberOfAtoms==5:
        index = np.random.choice(Pt5_index,p=Pt5_prob)
        assignedRadius = Pt5_data[index][0]
        assignedEnergy = Pt5_data[index][1]
        return assignedRadius, assignedEnergy
    if numberOfAtoms==6:
        index = np.random.choice(Pt6_index,p=Pt6_prob)
        assignedRadius = Pt6_data[index][0]
        assignedEnergy = Pt6_data[index][1]
        return assignedRadius, assignedEnergy
    if numberOfAtoms==7:
        index = np.random.choice(Pt7_index,p=Pt7_prob)
        assignedRadius = Pt7_data[index][0]
        assignedEnergy = Pt7_data[index][1]
        return assignedRadius, assignedEnergy
    if numberOfAtoms==8:
        index = np.random.choice(Pt8_index,p=Pt8_prob)
        assignedRadius = Pt8_data[index][0]
        assignedEnergy = Pt8_data[index][1]
        return assignedRadius, assignedEnergy

    return assignedRadius, assignedEnergy

# Metropolis loop begins here
np.random.seed()  # seed for random number generator
start_time_MC   = timeit.default_timer()
OUTPUT_data  = copy.deepcopy(INIT_data)
PES_copy     = copy.deepcopy(PES)

# Create LOG file
with open('LOG', 'w') as f5:
    f5.write('%s\n' % ('**********LOG info**********'))
    f5.write('\n')

# Identify largest cluster 
LCG = 0
for cluster in OUTPUT_data:
    if LCG <= cluster[1]:
        LCG =  cluster[1]

# Create and begin writing output to metropolis file
with open('metropolis','w') as f4:
    for step in range(Metro_Max+1):
        
        # Mark the beginning of loop
        with open('LOG', 'a') as f5:
            f5.write('%10s%0i\n' % ('Begin step ',step))
            f5.write('-' * len('Begin step ' + str(step)) + '\n')

        # Count number of atoms
        totalAtoms = 0
        for tar in OUTPUT_data:
            totalAtoms = totalAtoms + tar[0]
        #print(step)
        #print(str(step)+ ": " + str(totalAtoms))
        
        # Combine any overlapping clusters at the beginning of each metropolis loop
        overlapNumberCount = 0
        indexListAll = []
        ovlp = True
        while (ovlp and (overlapNumberCount < param.LimitForOverlap)):
            
            OUTPUT_data, ovlp, index_list, LCG = overlap_check(Clusters, OUTPUT_data, LCG)
            totalAtoms1 = 0
            for tar in OUTPUT_data:
                totalAtoms1 = totalAtoms1 + tar[0]
           # print(str(overlapNumberCount)+ ": " + str(totalAtoms1))
            Ncluster = len(OUTPUT_data)
           
            for index in index_list:
                indexListAll.append(index)

            overlapNumberCount += 1

        # Record any overlaps to LOG
        if indexListAll != []:
            with open('LOG', 'a') as f5:
                f5.write('%27s \n' %  ('**********OVERLAP**********'))
                f5.write('%27s ' %  ('Overlapping clusters found: '))
                indices_str = ', '.join(str(lst) for lst in indexListAll)
                f5.write(indices_str + '\n')
                f5.write('\n')
            #raise ValueError('Overlapping clusters found in the initial setup!')

        # Write current cluster configuration to metropolis
        if ( (step % write_step) == 0 ): 
            f4.write('%5s  %10i %16s %3i \n' % ('step =',step,'numclusters =',Ncluster))
            f4.write('%4s  %14s  %14s  %14s  %14s\n' %  ('Pt', 'R', 'X','Y','E'))
            for i in range(Ncluster):
                f4.write('%3i  %16.8f  %16.8f  %16.8f  %16.8f\n' % (OUTPUT_data[i][0], OUTPUT_data[i][1], OUTPUT_data[i][2], OUTPUT_data[i][3], OUTPUT_data[i][4]))
            f4.write('\n')
        
        # Choose a cluster to sinter
        indx = int(np.random.rand() * Ncluster) 
        num_atm_temp  = OUTPUT_data[indx][0] 
        R_temp        = OUTPUT_data[indx][1]
        X_temp        = OUTPUT_data[indx][2]
        Y_temp        = OUTPUT_data[indx][3]
        E_temp        = OUTPUT_data[indx][4]
        
        # Choose step size  
        if ( num_atm_temp == 1 ): #For monomer case.  
            px = int( 4.0*(2.0*np.random.rand() - 1.0) )   # -4 < px < 4   
            py = int( 4.0*(2.0*np.random.rand() - 1.0) )   # -4 < py < 4   
        else:  #If part of cluster we want bigger step, otherwise never breaks. We ensure that 
            px = math.ceil( ( 2.0*R_temp)  * ( 2.0*np.random.rand() - 1.0 ) ) # the step-size is cluster size dependent
            py = math.ceil( ( 2.0*R_temp)  * ( 2.0*np.random.rand() - 1.0 ) ) 
        X_new = X_temp + px * param.xstep_max
        Y_new = Y_temp + py * param.ystep_max
        
        if ( X_new == X_temp and Y_new == Y_temp ):
            continue
        
        # Enforce periodic Boundry Conditions  
        if ( X_new > param.maxx ):   
            X_new = param.minx - param.maxx + X_new
        elif ( X_new < param.minx ):   
            X_new = param.maxx - param.minx + X_new
        if ( Y_new > param.maxy ):  
            Y_new = param.miny - param.maxy + Y_new
        elif ( Y_new < param.miny ): 
            Y_new = param.maxy - param.miny + Y_new
        
        # Check if the moved monomer has landed in a existing cluster 
        inside_cluster = False
        for i in range (Ncluster):
            # if it is inside a cluster. the cluster can be just one atom
            temp_dist = distance(X_new, Y_new, OUTPUT_data[i][2], OUTPUT_data[i][3])
            temp_r    = OUTPUT_data[i][1] + param.Ratom
            if ( (temp_dist <  temp_r) and (i != indx) ):# i != indx means that it cannot be itself.   
                Eold = E_temp + OUTPUT_data[i][4]       # total Eold
                
                if ( OUTPUT_data[indx][0] == 1):# to check whether it is an atom or not
                    Enew_rmv = 0.0
                    Enew_add, Rnew_add = Cluster_finder(Clusters, OUTPUT_data, i, i, 'add')
               
                elif ( OUTPUT_data[indx][0] == 2):# ??? why the case of two atoms is special?
                    # CLUSTER FINDER FOR EADD
                    Enew_add, Rnew_add = Cluster_finder(Clusters, OUTPUT_data, i, i, 'add')
                    Enew_rmv = PES_finder(X_new, Y_new, PES_copy, N_mesh)
                    Rnew_rmv = param.Ratom
                # choose a new cluster

                else:
                    Enew_add, Enew_rmv, Rnew_add, Rnew_rmv = Cluster_finder(Clusters, OUTPUT_data, indx, i, 'both')
                Enew = Enew_add + Enew_rmv

                # Apply Metropolis condition here
                cond1 = (Enew-Eold < 0.0 and E_temp != OUTPUT_data[i][4])
                cond2 = (E_temp == OUTPUT_data[i][4] and np.exp( -beta*abs(Enew-Eold)) / (np.pi*OUTPUT_data[i][0]**2)  > np.random.rand())
                cond3 = (E_temp != OUTPUT_data[i][4] and np.exp( -beta*(Enew-Eold) ) > np.random.rand())
                
                if cond1 or cond2 or cond3: 
                # modify old clusters 
                    OUTPUT_data[i][0] = OUTPUT_data[i][0] + 1     # Natom
                    OUTPUT_data[i][1] = Rnew_add                     # R
                    OUTPUT_data[i][4] = Enew_add                     # E
                    if ( OUTPUT_data[indx][0] == 1 ):        #Natom = 1
                        # DELETE A CLUSTER (single atom)
                        del (OUTPUT_data[indx])
                        Ncluster -= 1
                    else:
                        OUTPUT_data[indx][0] = OUTPUT_data[indx][0] - 1     # Natom
                        OUTPUT_data[indx][1] = Rnew_rmv                        # R
                        OUTPUT_data[indx][4] = Enew_rmv                        # E

                else:
                    with open('LOG', 'a') as f5:
                        f5.write('%s \n' %  ('**********MOVE**********'))
                        f5.write('%s \n' %  ('METROPOLIS MOVE TO A NEW CLUSTER IS NOT FAVORABLE!'))
                inside_cluster = True
                break  
 
        if( inside_cluster == False ):  # we get a new single atom meaing that a new cluster forms      
            #print('inside cluster = false')
            if ( OUTPUT_data[indx][0] == 1):
                Enew_rmv = 0.0
            elif ( OUTPUT_data[indx][0] == 2):
                Enew_rmv = PES_finder(X_temp, Y_temp, PES_copy, N_mesh)
                Rnew_rmv = param.Ratom
            # choose a new cluster 
       
            else:
                Enew_rmv, Rnew_rmv = Cluster_finder(Clusters, OUTPUT_data, indx, indx, 'remove')
            E_atom = PES_finder(X_new, Y_new, PES_copy, N_mesh)
            Enew   = E_atom + Enew_rmv 
            Eold   = OUTPUT_data[indx][4]

            if ( (Enew-Eold < 0.0) or ( np.exp( -beta*(Enew-Eold) ) > np.random.rand()) ):  # accept the move
                with open('LOG', 'a') as f5:
                    f5.write('%s \n' %  ('**********MOVE**********'))
                    f5.write('%s \n' %  ('SINGLE ATOM MOVE OUTSIDE OF A CLUSTER ACCEPTED!'))
                # modify old clusters  
                if ( OUTPUT_data[indx][0] == 1 ):     # Note for a single atom move on surface
                    del (OUTPUT_data[indx])
                    Ncluster -= 1
                else:
                    OUTPUT_data[indx][0] = OUTPUT_data[indx][0] - 1     # Natom
                    OUTPUT_data[indx][1] = Rnew_rmv                        # R
                    OUTPUT_data[indx][4] = Enew_rmv                        # E
                OUTPUT_data.append([1, param.Ratom, X_new, Y_new, E_atom])             
                Ncluster += 1      
            else:
                with open('LOG', 'a') as f5:
                    f5.write('%s \n' %  ('**********MOVE**********'))
                    f5.write('%s \n' %  ('SINGLE ATOM MOVE OUTSIDE OF THE CLUSTER IS NOT ACCEPTED!'))

        # Mark the end of the loop
        with open('LOG', 'a') as f5:
            f5.write('\n')

with open('LOG', 'a') as f5:
    f5.write('\n')
    f5.write('%s \n' %  ('**********CALCULATION IS DONE**********'))
    f5.write('%13s  %12i\n' % ('Total steps =',step))
    f5.write('%30s %4i \n' %  ('Final number of clusters =', Ncluster))
    f5.write('%30s %16.8f \n' %  ('Total MC time in seconds =', timeit.default_timer() - start_time_MC))
    f5.write('%5s \n' %  ('DONE!'))
    f5.write('%s \n' %  ('***************************************'))

if param.SinteringResultPlot:
        # plot setting

    fig,axs = plt.subplots(1,2,sharey=False) # Defines ax variable by creating an empty plot

# Configure Both Subplots
    axs[0].grid(which='both', axis='both', linestyle='--')
    axs[1].grid(which='both', axis='both', linestyle='--')
    axs[0].set_axisbelow(True)
    axs[1].set_axisbelow(True)

    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].xaxis.set_major_locator(MultipleLocator(10))
    axs[1].xaxis.set_minor_locator(MultipleLocator(2))
    axs[1].yaxis.set_major_locator(MultipleLocator(10))
    axs[1].yaxis.set_minor_locator(MultipleLocator(2))
    axs[1].set_ylim(0,param.maxy)
    axs[1].set_xlim(0,param.maxx)
    axs[1].set_title('Final Distribution')

    axs[0].tick_params(axis='both', labelsize=14)
    axs[0].xaxis.set_major_locator(MultipleLocator(10))
    axs[0].xaxis.set_minor_locator(MultipleLocator(2))
    axs[0].yaxis.set_major_locator(MultipleLocator(10))
    axs[0].yaxis.set_minor_locator(MultipleLocator(2))
    axs[0].set_ylim(0,param.maxy)
    axs[0].set_xlim(0,param.maxx)
    axs[0].set_title('Initial Distribution')

    xcoords = []
    ycoords = []
    Rcoords = []
    types   = []
    for i in range(len(INIT_data)):
        xcoords.append(INIT_data[i][2])
        ycoords.append(INIT_data[i][3])
        Rcoords.append(30*int(INIT_data[i][1])**2)
        types.append(int(INIT_data[i][0]))

    for i,type in enumerate(types):
        x = xcoords[i]
        y = ycoords[i]
        axs[0].scatter(xcoords, ycoords, color='blue', s=Rcoords, marker='o')
        axs[0].text(x, y, type, fontsize=(12+type/3), color='yellow', horizontalalignment='center',
                 verticalalignment='center')

    xcoords = []
    ycoords = []
    Rcoords = []
    types   = []
    for i in range(len(OUTPUT_data)):
        xcoords.append(OUTPUT_data[i][2])
        ycoords.append(OUTPUT_data[i][3])
        Rcoords.append(30*int(OUTPUT_data[i][1])**2)
        types.append(int(OUTPUT_data[i][0]))

    for i,type in enumerate(types):
        x = xcoords[i]
        y = ycoords[i]
        axs[1].scatter(xcoords, ycoords, color='blue', s=Rcoords, marker='o')
        axs[1].text(x, y, type, fontsize=(12+type/3), color='yellow', horizontalalignment='center',
                 verticalalignment='center')

    plt.show()

