# get the relevant dependencies 
import json
import os
import sys 
import numpy as np
import inspect
import os.path
from math import pi, atan, sin, tanh

# fileimport that should always work when executing this file
filename = inspect.getframeinfo(inspect.currentframe()).filename
file_path = os.path.dirname(os.path.abspath(filename))
project_root = os.path.dirname(file_path)

if project_root not in sys.path:
    sys.path.append(project_root)
from event_model import event_model as em
import data_analysis.event_generator as eg

#### Interesting Implementations
# https://github.com/andreasfelix/hopfieldnetwork/tree/9e06c43c71c858afe4d8fc74f4c11c63ef83c22c



class smallHopfieldNetwork():
    # takes 3 modules
    def __init__(self, m1, m2, m3, alpha=1, beta=10, gamma=10) -> None:
        # m for module
        self.m1_hits = m1.hits()
        self.m2_hits = m2.hits()
        self.m3_hits = m3.hits()

        # c for count of hits in module i
        self.c1 = len(self.m1_hits)
        self.c2 = len(self.m2_hits)
        self.c3 = len(self.m3_hits)

        self.extracted_tracks = None

        self.setup_neurons(uniform_initialization=False)
        self.init_weights(alpha=alpha, beta=beta, gamma=gamma)

    def setup_neurons(self, uniform_initialization=True):
        # how are these orderd:
        # N1: i propose in the order all hit_1_0 -> all hit_2_i then hit_1_1 -> all hit_2_i, etc... 
        # N2: hit_2_0 to all hit_3_i then hit_2_1 to all hit_3_i, etc...
        # TODO agreable?
        if uniform_initialization:
            self.N1 = np.random.uniform(size = (self.c1 * self.c2, 1)) # all connections between m1 & m2 -> dtype np.array size |m1| * |m2| 
            self.N2 = np.random.uniform(size = (self.c2 * self.c3, 1)) # all connections between m2 & m3 -> dtype np.array size |m1| * |m2|
        else:
            self.N1 = np.ones(shape = (self.c1 * self.c2, 1)) # all connections between m1 & m2 -> dtype np.array size |m1| * |m2| 
            self.N2 = np.ones(shape = (self.c2 * self.c3, 1)) # all connections between m2 & m3 -> dtype np.array size |m1| * |m2|
   
        # Passaleva Initializes all neurons as "switched on" and not randomly (p. 869)
        # So we might try to set them to 1 or another 'activated' state


        # this is where we describe the angles in 3d space
        # I suppose lets go with the angles on zx and zy plane to describe the ordering of the direction in space (alt store a vector form one point to the other)
        # angles undirected so 0 - 180deg or 0 to pi 
        self.N1_info = np.zeros(shape = (self.c1 * self.c2, 2)) 
        self.N2_info = np.zeros(shape = (self.c2 * self.c3, 2))

        for i, hit1 in enumerate(self.m1_hits):
            for j, hit2 in enumerate(self.m2_hits):
                n_idx = i * self.c2 + j
                # this calculates angles of the line projections of the line segment defined by the hitpoints on zx, zy planes  
                angle_xz = atan((hit2.x - hit1.x)/(hit2.z - hit1.z))
                angle_yz = atan((hit2.y - hit1.y)/(hit2.z - hit1.z))

                self.N1_info[n_idx, 0] = angle_xz
                self.N1_info[n_idx, 1] = angle_yz
        
        for i, hit1 in enumerate(self.m2_hits):
            for j, hit2 in enumerate(self.m3_hits):
                n_idx = i * self.c3 + j
                # this calculates angles of the line projections of the line segment defined by the hitpoints on zx, zy planes  
                angle_xz = atan((hit2.x - hit1.x)/(hit2.z - hit1.z))
                angle_yz = atan((hit2.y - hit1.y)/(hit2.z - hit1.z))

                self.N2_info[n_idx, 0] = angle_xz
                self.N2_info[n_idx, 1] = angle_yz
                

    # fill the weight matrix based on the geometric properties of the hits / segments
    def init_weights(self, alpha=1, beta=10, gamma=10):
        # its a bit like looking at small summatrices that need weights: |m1| segments of N1 far connected to |m2| segemtns in N2 
        self.weights = np.zeros(shape = (self.N1.shape[0], self.N2.shape[0])) 
        # i will init the weights as the angels between the lines -> just to see make the logic which ones to 
        # important here t handle the weights for segments that dont share a hit accordingly

        # con representing the idx of the hit in m2 that connects the segements in N1 and N2
        # intuitively i need to calc weights for all segments that connect via a point in m2
        for con in range(self.c2):
            # go over all segments (coming from hits in m1) that connect to the current con 
            for i in range(self.c1):
                n1_idx = i * self.c2 + con 
                # go over all segments coming from 'con' that connect to all hits in m3
                for j in range(self.c3): 
                    n2_idx = con * self.c3 + j
                    # this is the angle difference between the projected line segments in zx
                    theta = abs(self.N1_info[n1_idx,0] - self.N2_info[n2_idx,0])
                    phi = abs(self.N1_info[n1_idx,1] - self.N2_info[n2_idx,1])
                    self.weights[n1_idx, n2_idx] = alpha * ((1-sin(theta))**beta) * ((1-sin(phi))**gamma)
    
    def energy(self, B = 1):
        # test calculation for the energy using the matrices 
        # bifurc penalty term (i.e. if two segments go from 2 hits in m1 to one hit in m2)
        # to do this easily lets reshape the neurons
        N1_pen = self.N1.reshape(self.c2, self.c1)
        N2_pen = self.N2.reshape(self.c2, self.c3)
        bifurc_pen = np.sum(np.trace(N1_pen @ N1_pen.T)) + np.sum(np.trace(N2_pen @ N2_pen.T)) - \
            np.sum(self.N1*self.N1) - np.sum(self.N2 * self.N2) 
        
         # dim -> (1,n1) x (n1,n2) x (n2,1)
        E = -0.5 * (self.N1.T @ self.weights @ self.N2) + B * bifurc_pen

        return E[0][0]
        
    def update(self, B = 1, T = 1):
        # after applying the update rule we need to check that the neurons are in the bound 0, 1 and if not set them to it
        # first we update the N1 neurons then the N2 neurons
       
        # N1 neurons
        for i in range(self.c1 * self.c2):
            # -> update is basically the sum of the active n2 neurons times the weights
            update = self.weights[i,:] @ self.N2

            # all neurons going into hit in m2 + value of all neurons going out of m2 
            # we look at neurons of segm between N1 and N2 and we want to 
            m1_id = i // self.c2
            m2_id = i % self.c2 

            # all segments mapping to the hit in m1
            m1h = np.sum(self.N1[m1_id * self.c2: (m1_id+1) * self.c2])
            
            # all segments mapping to the hit in m2
            m2h = np.sum(self.N1.reshape(self.c2, self.c1)[m2_id, :])

            # we need to substact the neuron of the segment 2 times because we add it 2 times
            pen =  m1h + m2h - 2 * self.N1[i] 

            self.N1[i] = 0.5 * (1 + tanh(update/(T) - B*pen/(T)))

        # N2 neurons
        for i in range(self.c2 * self.c3):
            # -> update is basically the sum of the active n2 neurons times the weights
            update = self.N1.T @ self.weights[:,i]
            
            m2_id = i // self.c3
            m3_id = i % self.c3 

            # all segments mapping to the hit in m2
            m2h = np.sum(self.N2[m2_id * self.c3: (m2_id+1) * self.c3])
            
            # all segments mapping to the hit in m3
            m3h = np.sum(self.N2.reshape(self.c3, self.c2)[m3_id, :])

            # we need to substact the neuron of the segment 2 times because we add it 2 times
            pen =  m2h + m3h - 2 * self.N2[i] 
            self.N2[i] = 0.5 * (1 + tanh(update / T - B * pen / T))
        # I think it also makes sense to process the neurons in blocks of neurons that go into the same hit

    def converge(self, B=1, T=5, convergence_threshold=0.0005):
        # Basically keep updating until the difference in Energy between timesteps is lower than 0.0005 (Based on Stimfple-Abele)
        # Passaleva uses a different kind of convergence i think (4)
        energies = [self.energy(B=B)] # store all energies (not fastest but maybe nice for visualisations)
        self.update(B=B, T=T)
        energies.append(self.energy(B=B))
        t = 0 # timesteps
        while abs(abs(energies[-2]) - abs(energies[-1])) >= convergence_threshold:
            self.update(B=B, T=5)
            energies.append(self.energy(B=B))
            t += 1
        
        self.update(B=B, T=5)
        energies.append(self.energy(B=B))

        print("Network Converged after " + str(t) + " steps")
        print("Energy = " + str(energies[-1]))


    def tracks(self, activation_threshold : list=None, max_activation=False):
        if self.extracted_tracks:
            return self.extracted_tracks
        print("Starting extraction...")
        # What the papers say:  The answer is given by the final set of active Neurons
        #                       All sets of Neurons connected together are considered as track candidates
        #                       
        # IDEA: All neurons that share a hit and are both connected are track candidates
        if activation_threshold is None:
            activation_threshold = [0, 0]
        candidates = []
        # for con in range(self.c2):
        #     # go over all segments (coming from hits in m1) that connect to the current con
        #     for i in range(self.c1):
        #         n1_idx = i * self.c2 + con
        #         # go over all segments coming from 'con' that connect to all hits in m3
        #         for j in range(self.c3):
        #             n2_idx = con * self.c3 + j
        #
        #             # if both are over the activaten threshold the two segments are a track candidate
        #             if self.N1[n1_idx] > activation_threshold and self.N2[n2_idx] > activation_threshold:
        #                 candidates.append((n1_idx, n2_idx))

        # New method for computing the tracks (using above concepts) --------------------------

        l1 = self.c1  # number of hits in module 1
        l2 = self.c2
        l3 = self.c3
        for i, state1 in enumerate(self.N1):
            for j, state2 in enumerate(self.N2):
                if state1 < activation_threshold[0] or state2 < activation_threshold[1]:
                    continue
                # segments are connected via hit in m2
                if int(i % l1) == int(j // l3):
                    candidates.append(em.track([self.m1_hits[int(i // l2)],
                                                self.m2_hits[int(i % l1)],
                                                self.m3_hits[int(j % l2)]]))

        if max_activation:
            candidates = []
            n1_transform = self.N1.reshape(self.c2, self.c1)
            for con in range(self.c2):  # loop over the connection hits in module 2
                h1_idx = np.argmax(n1_transform[con, :])
                h3_idx = np.argmax(self.N2[con*self.c3:(con+1)*self.c3])
                candidates.append(em.track([self.m1_hits[h1_idx],
                                            self.m2_hits[con],
                                            self.m3_hits[h3_idx]]))
                print(f"{round(con/self.c2*100,2)}% done extracting...")
        self.extracted_tracks = candidates

        return candidates

    # Just a function to print some stats about the network
    def network_stats(self, activation_threshold=0):
        print("+-------------------------------------------------------+")
        print("| Network Stats ")
        print("+----------------------+--------------------------------+")
        print("| # Neurons            | " + str(len(self.N1) + len(self.N2)))
        print("+----------------------+--------------------------------+")
        print("| N1 activation mean:  | " + str(np.mean(self.N1)))
        print("+----------------------+--------------------------------+")
        print("| # N1 > act_threshold | " + str(len(self.N1[self.N1 > activation_threshold])))
        print("+----------------------+--------------------------------+")
        print("| # N1 <= act_threshold| " + str(len(self.N1[self.N1 <= activation_threshold])))
        print("+----------------------+--------------------------------+")
        print("| N2 activation mean:  | " + str(np.mean(self.N2)))
        print("+----------------------+--------------------------------+")
        print("| # N2 > act_threshold | " + str(len(self.N2[self.N2 > activation_threshold])))
        print("+----------------------+--------------------------------+")
        print("| # N2 <= act_threshold| " + str(len(self.N2[self.N2 <= activation_threshold])))
        print("+----------------------+--------------------------------+")

    def plot_network_result(self, activation_threshold=None, max_segment=False):
        if activation_threshold:
            t = self.tracks(activation_threshold=activation_threshold)
        elif max_segment:
            t = self.tracks(max_activation=True)
        else:
            t = self.tracks()
        eg.plot_tracks_and_modules(t, [m1, m2, m3], title="Hopfield Output")


if __name__ == "__main__":
    # loading some event
    # event_path = os.path.join(project_root, "events/bsphiphi/velo_event_12.json")
    # f = open(event_path)
    # json_data = json.loads(f.read())
    # event = em.event(json_data)

    # taking a subsection of the event ( 3 modules to feed into the small hopfield net)
    # lets only take the first 3 even modules
    
    #################### PARAMETERS #######################
    ### WEIGHTS ###

    ALPHA = 1
    BETA = 10
    GAMMA = 10

    #######################################################

    # Generating a test event to work with
    tracks = []
    while len(tracks) == 0:
        tracks = eg.generate_test_tracks(allowed_modules=[0, 2, 4], num_test_events=1,
                                         num_tracks=50, reconstructable_tracks=True,
                                         random_pool_size=20, dataset="minibias")[0]
    modules = eg.tracks_to_modules(tracks)
    eg.plot_tracks_and_modules(tracks, modules, title="Generated Instance")

    m1 = modules[0]
    m2 = modules[1]
    m3 = modules[2]
    my_hopfield = smallHopfieldNetwork(m1=m1, m2=m2, m3=m3, alpha=ALPHA, beta=BETA, gamma=GAMMA)
    my_hopfield.network_stats()
    my_hopfield.converge()
    my_hopfield.network_stats()
    my_hopfield.plot_network_result(max_segment=True)
    print(my_hopfield.tracks())
