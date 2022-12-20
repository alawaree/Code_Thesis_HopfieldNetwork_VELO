import hashlib
from math import sqrt, acos

class event(object):
  '''Event defined by its json description.'''
  def __init__(self, json_data, read_tracks=False):
    self.number_of_modules = 52
    self.description = json_data["description"]
    self.montecarlo = json_data["montecarlo"]
    self.module_prefix_sum = json_data["module_prefix_sum"]
    self.number_of_hits = self.module_prefix_sum[self.number_of_modules]
    self.module_zs = []
    self.hits = []
    with_t = "t" in json_data

    for m in range(self.number_of_modules):
      self.module_zs.append(set([]))
      for i in range(self.module_prefix_sum[m], self.module_prefix_sum[m + 1]):
        if with_t:
          self.hits.append(hit(json_data["x"][i], json_data["y"][i], json_data["z"][i], i, m, json_data["t"][i], 1))
        else:
          self.hits.append(hit(json_data["x"][i], json_data["y"][i], json_data["z"][i], i, m))
        self.module_zs[m].add(json_data["z"][i])
    
    self.modules = [
      module(m,
        self.module_zs[m],
        self.module_prefix_sum[m],
        self.module_prefix_sum[m + 1] - self.module_prefix_sum[m],
        self.hits
      ) for m in range(0, self.number_of_modules)
    ]

    self.real_tracks = []
    if read_tracks:
      for part in self.montecarlo.get('particles'):
        track_hits = part[15]
        self.real_tracks.append(track([hit for hit in self.hits if hit.id in track_hits]))

class track(object):
  '''A track, essentially a list of hits.'''
  def __init__(self, hits):
    self.hits = hits
    self.missed_last_module = False
    self.missed_penultimate_module = False
    self.min_r = None
    self.max_phi_change = 0
    # might consider consecutive
    self.direction = 0
    self.max_cosecutive_phi_change = 0
    self.first_hit_module = None

  def add_hit(self, hit):
    self.hits.append(hit)

  def __repr__(self):
    return "Track with " + str(len(self.hits)) + " hits: " + str(self.hits)

  def __iter__(self):
    return iter(self.hits)

  def __eq__(self, other):
    return self.hits == other.hits

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
      return int.from_bytes(hashlib.sha256(
        ''.join([str(h.id) for h in self.hits]).encode('utf-8')).digest(), byteorder='big')
 
  def update_track(self):
   # I simplify here! ignoring tracks like decays that can potentially end closer to the beam...
    if self.hits[0].pol_r < self.hits[-1].pol_r:
      self.min_r = self.hits[0].pol_r
      direction = 1
      self.first_hit_module = self.hits[0].module_number
    else:
      self.min_r = self.hits[-1].pol_r
      direction = -1
      self.first_hit_module = self.hits[-1].module_number

    last_phi = self.hits[0].pol_phi
    last_module = self.hits[0].module_number
    max_delta_phi = 0
    max_consec_delta_phi = 0
    for i,hit in enumerate(self.hits):
      hit.direction = direction
      if abs(last_phi-hit.pol_phi) > max_delta_phi:
        max_delta_phi = abs(last_phi-hit.pol_phi)
      if abs(last_phi-hit.pol_phi) > max_consec_delta_phi and (hit.module_number - last_module) < 3:
        max_consec_delta_phi = abs(last_phi-hit.pol_phi)
      last_module = hit.module_number
      last_phi = hit.pol_phi
  
    self.max_phi_change = max_delta_phi
    # might consider consecutive
    self.max_cosecutive_phi_change = max_consec_delta_phi
    self.direction = direction

class hit(object):
  '''A hit, composed of an id and its x, y and z coordinates.
  It may optionally contain the number of the module where
  the hit happened.
  '''
  def __init__(self, x, y, z, hit_id, module=-1, t=0, with_t=False, direction=0):
    self.x = x
    self.y = y
    self.z = z
    self.t = t
    self.id = hit_id
    self.module_number = module
    self.with_t = with_t
    self.direction = direction
    self.pol_phi = None
    self.pol_r = None

  def __getitem__(self, index):
    if (index<0 or index>2):
      raise IndexError

    if (index==0): return self.x
    elif(index==1): return self.y
    else: return self.z

  def __repr__(self):
    return "#" + str(self.id) + " module " + str(self.module_number) + " {" + str(self.x) + ", " + \
         str(self.y) + ", " + str(self.z) + (", " + str(self.t) if self.with_t else "") + "}"

  def __eq__(self, other):
      return self.id == other.id and self.x == other.x and self.y == other.y and self.z == other.z

  def __ne__(self, other):
      return not self.__eq__(other)

  def __hash__(self):
      return self.id
    
  def update_polar(self):
    self.pol_r = sqrt(self.x**2 + self.y**2)
    self.pol_phi = acos(self.x/self.pol_r)
    if self.y < 0:
      self.pol_phi *= -1

class module(object):
  '''A module is identified by its number.
  It also contains the z coordinate in which it sits, and
  the list of hits it holds.

  Note modules are ordered by z, so the less the module_number,
  the less the z.
  '''
  def __init__(self, module_number, z, start_hit, number_of_hits, hits):
    self.module_number = int(module_number)
    self.z = z
    self.hit_start_index = start_hit
    self.hit_end_index = start_hit + number_of_hits
    self.__global_hits = hits

  def __iter__(self):
    return iter(self.__global_hits[self.hit_start_index : self.hit_end_index])

  def __repr__(self):
    return "module " + str(self.module_number) + ":\n" + \
      " At z: " + str(self.z) + "\n" + \
      " Number of hits: " + str(len(self.hits())) + "\n" + \
      " Hits (#id {x, y, z}): " + str(self.hits())

  def hits(self):
    return self.__global_hits[self.hit_start_index : self.hit_end_index]
